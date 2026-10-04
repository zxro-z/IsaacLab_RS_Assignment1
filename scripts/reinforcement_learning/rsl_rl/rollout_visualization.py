"""Opt-in presentation only: no physics, policy, observation or episode changes."""

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from pxr import Gf, UsdGeom, UsdLux, UsdShade, Sdf


class FirstEpisodeReliefVideo(gym.wrappers.RecordVideo):
    """Hold the last pre-terminal image for one frame instead of filming auto-reset.

    The terminal transition still occurs normally and is logged by telemetry.
    Only the encoder's frame list changes; no additional render or physics step.
    """

    def _capture_frame(self):
        if bool(self.env.unwrapped.termination_manager.dones[0]) and self.recorded_frames:
            self.recorded_frames.append(self.recorded_frames[-1].copy())
        else:
            super()._capture_frame()


class TerrainReliefVisualization:
    EYE = (-3.8, -3.2, 1.8)
    LOOKAT = (1.0, 0.0, -0.15)

    def __init__(self, env):
        self.env = env
        self.stage = env.scene.stage
        self.before = self.physical_signature()
        material = UsdShade.Material.Define(self.stage, "/World/QualitativeTerrainMaterial")
        shader = UsdShade.Shader.Define(self.stage, "/World/QualitativeTerrainMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.48, 0.50, 0.52))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mesh = self.stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path + "/terrain/mesh")
        # Visual binding leaves the explicit physics-purpose binding intact.
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        light = UsdLux.DistantLight(self.stage.GetPrimAtPath("/World/light"))
        light.CreateIntensityAttr(3000.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
        light.CreateAngleAttr(0.5)
        xf = UsdGeom.Xformable(light.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, -78.0, -35.0))
        fill = UsdLux.DomeLight.Define(self.stage, "/World/QualitativeFill")
        fill.CreateIntensityAttr(120.0)
        fill.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.0))
        assert self.before == self.physical_signature(), "Rendering changed a physical signature"

    def physical_signature(self):
        prim = self.stage.GetPrimAtPath(self.env.cfg.scene.terrain.prim_path + "/terrain/mesh")
        mesh = UsdGeom.Mesh(prim)
        h = hashlib.sha256()
        h.update(np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32).tobytes())
        h.update(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).tobytes())
        bindings, physics = {}, {}
        for p in self.stage.Traverse():
            for rel in p.GetRelationships():
                if "material:binding:physics" in rel.GetName():
                    bindings[str(rel.GetPath())] = [str(t) for t in rel.GetTargets()]
            for attr in p.GetAttributes():
                if attr.GetName().startswith(("physics:staticFriction", "physics:dynamicFriction", "physics:restitution")):
                    physics[str(attr.GetPath())] = str(attr.Get())
        return {"terrain_mesh_sha256": h.hexdigest(), "physics_bindings": bindings,
                "physics_material_attributes": physics,
                "robot_material_sha256": hashlib.sha256(self.env.scene["robot"].root_physx_view.get_material_properties().cpu().numpy().tobytes()).hexdigest(),
                "env_origins_sha256": hashlib.sha256(self.env.scene.env_origins.cpu().numpy().tobytes()).hexdigest()}

    def update_camera(self):
        w, x, y, z = self.env.scene["robot"].data.root_quat_w[0].cpu().tolist()
        yaw = math.atan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
        c, s = math.cos(yaw), math.sin(yaw)
        rotate = lambda p: (c*p[0]-s*p[1], s*p[0]+c*p[1], p[2])
        self.env.viewport_camera_controller.update_view_location(eye=rotate(self.EYE), lookat=rotate(self.LOOKAT))

    def finalize(self, directory):
        after = self.physical_signature()
        assert self.before == after, "Physical signatures changed during recording"
        with (Path(directory) / "visualization.json").open("x") as f:
            json.dump({"camera_eye_heading_frame": self.EYE, "lookat_heading_frame": self.LOOKAT,
                       "camera_logic": "Yaw-only rotation from decision-time root quaternion; native asset-root translation; no roll/pitch rotation.",
                       "terrain_visual_diffuse": [0.48, 0.50, 0.52], "roughness": 0.9, "metallic": 0.0,
                       "light": {"intensity": 3000.0, "rotation_xyz_deg": [0, -78, -35], "color": [1, .96, .90], "angle": .5},
                       "dome_fill": {"intensity": 120.0, "color": [.85, .90, 1]},
                       "physical_signature_before": self.before, "physical_signature_after": after,
                       "physics_geometry_unchanged": True, "physics_material_unchanged": True,
                       "terminal_frame": "Hold previous frame for one 1/60-second frame; no reset image, no extra step."}, f, indent=2)
