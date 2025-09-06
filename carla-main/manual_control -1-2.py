#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) ...
# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
     or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from carla import VehicleLightState as vls
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import imageio
from openpyxl import Workbook
import time


try:
    import pygame
    from pygame.locals import (
        KMOD_CTRL, KMOD_SHIFT,
        K_0, K_9, K_BACKQUOTE, K_BACKSPACE,
        K_COMMA, K_DOWN, K_ESCAPE, K_F1,
        K_LEFT, K_PERIOD, K_RIGHT,
        K_SLASH, K_SPACE, K_TAB, K_UP,
        K_a, K_b, K_c, K_d, K_f, K_g,
        K_h, K_i, K_l, K_m, K_n, K_o,
        K_p, K_q, K_r, K_s, K_t, K_v,
        K_w, K_x, K_z, K_MINUS, K_EQUALS
    )
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(18)
driving_sound=pygame.mixer.Sound('driving_sound.wav')
channel_driving = pygame.mixer.Channel(0)
Car_horn_sound = pygame.mixer.Sound('Car_horn_sound_effect.wav')
channel_Car_horn_sound = pygame.mixer.Channel(1)
Car_reversing_warning_sound=pygame.mixer.Sound('Car_reversing_warning_sound.wav')
channel_Car_reversing_warning_sound = pygame.mixer.Channel(2)
blink_sound=pygame.mixer.Sound('blink.wav')
channel_blink = pygame.mixer.Channel(3)
car_start_sound=pygame.mixer.Sound('car_start.wav')
channel_car_start = pygame.mixer.Channel(4)
start_stop_sound=pygame.mixer.Sound('start_stop_sound.wav')
channel_start_stop = pygame.mixer.Channel(5)
car_idling=pygame.mixer.Sound('car-idling.wav')
channel_car_idling = pygame.mixer.Channel(7)
slow_moving=pygame.mixer.Sound('slow-moving-car.wav')
channel_slow_moving = pygame.mixer.Channel(8)
car_horn_beep=pygame.mixer.Sound('car-horn-beep.wav')
channel_car_horn_beep = pygame.mixer.Channel(6)
iddle=pygame.mixer.Sound('iddle.wav')
channel_iddle = pygame.mixer.Channel(10)
brake_sound=pygame.mixer.Sound('stop.wav')
channel_brake = pygame.mixer.Channel(11)
flip=pygame.mixer.Sound('flip.wav')
channel_flip = pygame.mixer.Channel(12)
flip_reverse=pygame.mixer.Sound('flip_reverse.wav')
channel_flip_reverse = pygame.mixer.Channel(13)
start_engine=pygame.mixer.Sound('start_engine.wav')
channel_start_engine = pygame.mixer.Channel(14)
end_sound=pygame.mixer.Sound('end_sound.wav')
channel_end_sound = pygame.mixer.Channel(15)
grade__sound=pygame.mixer.Sound('grade.wav')
channel_grade = pygame.mixer.Channel(16)

previous_lane_id = None
turning = False
turn_side = None
turn_initiation_heading = 0.0
previous_heading = None
steer_threshold = 0.3
last_left_blinker_on_time = -999.0
last_right_blinker_on_time = -999.0
image_number=0
start_stop=0
total_time=0
red_light=0
wrong_way_time=0
wrong_blinker=0
collision_times=0
Sudden_brake=0
highest_speed=0
cross_line_times=0
Honking_the_horn=0
speedly_time=0
end_count=0 
grade=100

def apply_mirror_mask(raw_surf, mask_surf):
    """Make the shape of the left rearview mirror"""
    raw_surf = raw_surf.convert_alpha()
    mask_surf = mask_surf.convert_alpha()
    final = pygame.Surface(raw_surf.get_size(), pygame.SRCALPHA, 32)
    final.blit(raw_surf, (0, 0))
    final.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return final

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all":
        return bps
    if len(bps) == 1:
        return bps
    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ======================================================================
#  World
# ======================================================================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.vehicle2=None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = 'vehicle.audi.A2'
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.mirror_camera = None
        self.mirror_surface = None
        self.left_mirror_camera = None
        self.left_mirror_surface = None
        self.side_mirror_mask = pygame.image.load("side_mirror_mask.png").convert_alpha()
        self.create_low_case_car()
        self.create_leave_car()
        self.create_bad_car()
        self.create_stage1_event1_car()
        self.create_stage2_event1_car()
        self.create_stage2_event2_car()
        self.create_stage2_event3_car()
        self.create_stage1_event1_car_2()
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ] 
        
    def create_leave_car(self):
        vehicle_bp_2 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_2 = carla.Location(x=168.2342, y=1.7040, z=0.2) 
        spawn_rotation_2 = carla.Rotation(pitch=0.0, yaw=270.0, roll=0)
        custom_transform_2 = carla.Transform(spawn_location_2, spawn_rotation_2)
        self.vehicle2 = self.world.try_spawn_actor(vehicle_bp_2, custom_transform_2)
        self.vehicle2.set_autopilot(False)
        self.vehicle2.disable_constant_velocity()
        self.modify_vehicle_physics(self.vehicle2)
        
        
    def create_bad_car(self):
        vehicle_bp_3 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_3 = carla.Location(x=198.00, y=-33.667, z=0.2)
        #spawn_location_3 = carla.Location(x=174.93, y=-7.344, z=0.2)
        spawn_rotation_3 = carla.Rotation(pitch=0.0, yaw=0, roll=0)
        custom_transform_3 = carla.Transform(spawn_location_3, spawn_rotation_3)
        self.vehicle3 = self.world.try_spawn_actor(vehicle_bp_3, custom_transform_3)
        self.vehicle3.set_autopilot(False)
        self.vehicle3.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle3) 
        
    def create_low_case_car(self):
        vehicle_bp_4 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_4 = carla.Location(x=201.30, y=19.164, z=0.2) 
        spawn_rotation_4 = carla.Rotation(pitch=0.0, yaw=-180, roll=0)
        custom_transform_4 = carla.Transform(spawn_location_4, spawn_rotation_4)
        self.vehicle4 = self.world.try_spawn_actor(vehicle_bp_4, custom_transform_4)
        self.vehicle4.set_autopilot(False)
        self.vehicle4.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle4)
    
    def create_stage2_event1_car(self):
        vehicle_bp_5 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_5 = carla.Location(x=180.34, y=-13.62, z=0.2) 
        spawn_rotation_5 = carla.Rotation(pitch=0.0, yaw=-90, roll=0)
        custom_transform_5 = carla.Transform(spawn_location_5, spawn_rotation_5)
        self.vehicle5 = self.world.try_spawn_actor(vehicle_bp_5, custom_transform_5)
        self.vehicle5.set_autopilot(False)
        self.vehicle5.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle5)
    
    def create_stage1_event1_car_2(self):
        vehicle_bp_9 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_9 = carla.Location(x=160.00, y=-33.667, z=0.2)
        #spawn_location_9 = carla.Location(x=196.92, y=30, z=0.2)
        spawn_rotation_9 = carla.Rotation(pitch=0.0, yaw=90, roll=0)
        custom_transform_9 = carla.Transform(spawn_location_9, spawn_rotation_9)
        self.vehicle9 = self.world.try_spawn_actor(vehicle_bp_9, custom_transform_9)
        self.vehicle9.set_autopilot(False)
        self.vehicle9.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle9)  
    
    def create_stage2_event2_car(self):
        vehicle_bp_6 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_6 = carla.Location(x=162.46, y=54.8, z=1) 
        spawn_rotation_6 = carla.Rotation(pitch=0.0, yaw=-90, roll=0)
        custom_transform_6 = carla.Transform(spawn_location_6, spawn_rotation_6)
        self.vehicle6 = self.world.try_spawn_actor(vehicle_bp_6, custom_transform_6)
        self.vehicle6.set_autopilot(False)
        self.vehicle6.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle6)
    
    def create_stage2_event3_car(self):
        vehicle_bp_7 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_7 = carla.Location(x=195.45, y=54.86, z=0.2) 
        spawn_rotation_7 = carla.Rotation(pitch=0.0, yaw=-90, roll=0)
        custom_transform_7 = carla.Transform(spawn_location_7, spawn_rotation_7)
        self.vehicle7 = self.world.try_spawn_actor(vehicle_bp_7, custom_transform_7)
        self.vehicle7.set_autopilot(False)
        self.vehicle7.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle7)
        
    def create_stage1_event1_car(self):
        vehicle_bp_8 = self.world.get_blueprint_library().find('vehicle.audi.tt')
        spawn_location_8 = carla.Location(x=201.41, y=46.41, z=0.2) 
        spawn_rotation_8 = carla.Rotation(pitch=0.0, yaw=-180, roll=0)
        custom_transform_8 = carla.Transform(spawn_location_8, spawn_rotation_8)
        self.vehicle8 = self.world.try_spawn_actor(vehicle_bp_8, custom_transform_8)
        self.vehicle8.set_autopilot(False)
        self.vehicle8.disable_constant_velocity() 
        self.modify_vehicle_physics(self.vehicle8)
                        
    def get_front_vehicle_distance(self):
        """Calculate the distance to the vehicle ahead"""
        ego_vehicle = self.player
        if not ego_vehicle:
            return None, None
        ego_location = ego_vehicle.get_transform().location
        ego_transform = ego_vehicle.get_transform()
        ego_waypoint = self.world.get_map().get_waypoint(ego_location)
        ego_lane_id = ego_waypoint.lane_id

        vehicles = self.world.get_actors().filter('vehicle.*')
        front_vehicle = None
        min_distance = float('inf')

        for v in vehicles:
            if v.id == ego_vehicle.id:
                continue
            loc = v.get_transform().location
            w = self.world.get_map().get_waypoint(loc)
            if not w:
                continue
            lane_id = w.lane_id
            if lane_id == ego_lane_id:
                relative_loc = loc - ego_location
                forward_dot = relative_loc.dot(ego_transform.get_forward_vector())
                if forward_dot > 0 and relative_loc.length() < 50:
                    dist = relative_loc.length()
                    if dist < min_distance:
                        min_distance = dist
                        front_vehicle = v

        return front_vehicle, min_distance if front_vehicle else None

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        cam_index = self.camera_manager.index if self.camera_manager else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager else 0

        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")

        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        if self.player is not None:
            sp = self.player.get_transform()
            sp.location.z += 2.0
            sp.rotation.roll = 0.0
            sp.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, sp)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            spawn_location = carla.Location(x=106.56, y=-21.588, z=0.2) 
            spawn_rotation = carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0)
            custom_transform = carla.Transform(spawn_location, spawn_rotation)
            self.player = self.world.try_spawn_actor(blueprint, custom_transform)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        mirror_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        mirror_bp.set_attribute('image_size_x', '230')
        mirror_bp.set_attribute('image_size_y', '55')
        mirror_bp.set_attribute('fov', '90')
        mirror_transform = carla.Transform(
            carla.Location(x=-1.8, z=0.8),
            carla.Rotation(yaw=180.0)
        )
        self.mirror_camera = self.world.try_spawn_actor(mirror_bp, mirror_transform, attach_to=self.player)
        if self.mirror_camera is not None:
            self.mirror_camera.listen(lambda img: self._on_mirror_image(img))

        left_mirror_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        left_mirror_bp.set_attribute('image_size_x', '150')
        left_mirror_bp.set_attribute('image_size_y', '100')
        left_mirror_bp.set_attribute('fov', '90')

        left_mirror_transform = carla.Transform(
            carla.Location(x=-0.1, y=-0.8, z=0.8),
            carla.Rotation(yaw=210.0, pitch=5.0)
        )
        self.left_mirror_camera = self.world.try_spawn_actor(left_mirror_bp, left_mirror_transform, attach_to=self.player)
        if self.left_mirror_camera is not None:
            self.left_mirror_camera.listen(lambda img: self._on_left_mirror_image(img))

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _on_mirror_image(self, image):
        """Center rear mirror image"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        surface = pygame.transform.flip(surface, True, False)
        self.mirror_surface = pygame.transform.scale(surface, (230, 55))
        

    def _on_left_mirror_image(self, image):
        """Left rearview mirror image"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        cam_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        cam_surface = cam_surface.convert_alpha()
        cam_surface = pygame.transform.scale(cam_surface, (145, 90)).convert_alpha()
        cam_surface = pygame.transform.flip(cam_surface, True, False).convert_alpha()
        mask_surf = pygame.transform.scale(self.side_mirror_mask, (145, 90)).convert_alpha()
        final_surf = apply_mirror_mask(cam_surface, mask_surf)
        self.left_mirror_surface = final_surf

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

        if self.mirror_surface is not None:
            display.blit(self.mirror_surface, (985, 145))

        if self.left_mirror_surface is not None:
            display.blit(self.left_mirror_surface, (142.5, 367.5))

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor
        ]
        for sensor in sensors:
            if sensor:
                sensor.stop()
                sensor.destroy()

        if self.mirror_camera is not None:
            self.mirror_camera.stop()
            self.mirror_camera.destroy()
            self.mirror_camera = None

        if self.left_mirror_camera is not None:
            self.left_mirror_camera.stop()
            self.left_mirror_camera.destroy()
            self.left_mirror_camera = None

        if self.player:
            self.player.destroy()

    def is_driving_against_traffic(self):
        ego_vehicle = self.player
        if not ego_vehicle:
            return False
        ego_location = ego_vehicle.get_transform().location
        ego_forward = ego_vehicle.get_transform().get_forward_vector()
        waypoint = self.world.get_map().get_waypoint(ego_location)
        lane_forward = waypoint.transform.get_forward_vector()

        dot_ = (
            ego_forward.x * lane_forward.x +
            ego_forward.y * lane_forward.y +
            ego_forward.z * lane_forward.z
        )
        mag_ego = math.sqrt(ego_forward.x**2 + ego_forward.y**2 + ego_forward.z**2)
        mag_lane = math.sqrt(lane_forward.x**2 + lane_forward.y**2 + lane_forward.z**2)

        angle = math.acos(dot_ / (mag_ego * mag_lane))
        return angle > math.pi / 2

    def get_lane_offset(self):
        """Calculate lane offset distance"""
        ego_vehicle = self.player
        if not ego_vehicle:
            return 0.0
        loc = ego_vehicle.get_transform().location
        waypoint = self.world.get_map().get_waypoint(loc)
        lane_center = waypoint.transform.location
        offset_vec = loc - lane_center
        forward_vec = ego_vehicle.get_transform().get_forward_vector()
        right_vec = carla.Vector3D(-forward_vec.y, forward_vec.x, 0)
        return offset_vec.dot(right_vec)
        
        

class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        self.hand_brake=0
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        global image_number
        global start_stop
        global Honking_the_horn
        global red_light
        """Key conversion events"""
        if isinstance(self._control, carla.VehicleControl):
                current_lights = self._lights
        if image_number>=5 and start_stop==1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                elif event.type == pygame.KEYUP:
                    if self._is_quit_shortcut(event.key):
                        return True
                    elif event.key == K_BACKSPACE:
                        if self._autopilot_enabled:
                            world.player.set_autopilot(False)
                            world.restart()
                            world.player.set_autopilot(True)
                        else:
                            world.restart()
                    elif event.key == K_F1:
                        world.hud.toggle_info()
                    elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                        world.next_map_layer(reverse=True)
                    elif event.key == K_v:
                        channel_car_horn_beep.play(car_horn_beep)
                        channel_car_horn_beep.set_volume(1)
                        Honking_the_horn+=1
                    elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                        world.load_map_layer(unload=True)
                    elif event.key == K_b:
                        world.load_map_layer()
                    elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                        world.hud.help.toggle()
                    elif event.key == K_TAB:
                        red_light+=1
                    elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                        world.next_weather(reverse=True)
                    elif event.key == K_c:
                            start_stop=0
                    elif event.key == K_g:
                        world.toggle_radar()
                    elif event.key == K_BACKQUOTE:
                        world.camera_manager.next_sensor()
                    elif event.key == K_n:
                        if self.hand_brake:
                            self.hand_brake=0
                        else:
                            self.hand_brake=1
                    elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                        if world.constant_velocity_enabled:
                            world.player.disable_constant_velocity()
                            world.constant_velocity_enabled = False
                            world.hud.notification("Disabled Constant Velocity Mode")
                        else:
                            world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                            world.constant_velocity_enabled = True
                            world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                    elif event.key == K_o:
                        try:
                            if world.doors_are_open:
                                world.hud.notification("Closing Doors")
                                world.doors_are_open = False
                                world.player.close_door(carla.VehicleDoor.All)
                            else:
                                world.hud.notification("Opening doors")
                                world.doors_are_open = True
                                world.player.open_door(carla.VehicleDoor.All)
                        except Exception:
                            pass
                    elif event.key == K_t:
                        if world.show_vehicle_telemetry:
                            world.player.show_debug_telemetry(False)
                            world.show_vehicle_telemetry = False
                            world.hud.notification("Disabled Vehicle Telemetry")
                        else:
                            try:
                                world.player.show_debug_telemetry(True)
                                world.show_vehicle_telemetry = True
                                world.hud.notification("Enabled Vehicle Telemetry")
                            except Exception:
                                pass
                    elif event.key > K_0 and event.key <= K_9:
                        index_ctrl = 0
                        if pygame.key.get_mods() & KMOD_CTRL:
                            index_ctrl = 9
                        world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                    elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                        world.camera_manager.toggle_recording()
                    elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                        if (world.recording_enabled):
                            client.stop_recorder()
                            world.recording_enabled = False
                            world.hud.notification("Recorder is OFF")
                        else:
                            client.start_recorder("manual_recording.rec")
                            world.recording_enabled = True
                            world.hud.notification("Recorder is ON")
                    elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                        client.stop_recorder()
                        world.recording_enabled = False
                        current_index = world.camera_manager.index
                        world.destroy_sensors()
                        self._autopilot_enabled = False
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification("Replaying file 'manual_recording.rec'")
                        client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                        world.camera_manager.set_sensor(current_index)
                    elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                        if pygame.key.get_mods() & KMOD_SHIFT:
                            world.recording_start -= 10
                        else:
                            world.recording_start -= 1
                        world.hud.notification("Recording start time is %d" % (world.recording_start))
                    elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                        if pygame.key.get_mods() & KMOD_SHIFT:
                            world.recording_start += 10
                        else:
                            world.recording_start += 1
                        world.hud.notification("Recording start time is %d" % (world.recording_start))

                    if isinstance(self._control, carla.VehicleControl):
                        if event.key == K_f:
                            self._ackermann_enabled = not self._ackermann_enabled
                            world.hud.show_ackermann_info(self._ackermann_enabled)
                            world.hud.notification("Ackermann Controller %s" % ("Enabled" if self._ackermann_enabled else "Disabled"))
                        if event.key == K_q:
                            if not self._ackermann_enabled:
                                if self._control.reverse:
                                    self._control.gear = 1
                                    channel_Car_reversing_warning_sound.fadeout(500)
                                else:
                                    self._control.gear = -1
                                    channel_Car_reversing_warning_sound.play(Car_reversing_warning_sound, loops=-1, fade_ms=500)
                            else:
                                self._ackermann_reverse *= -1
                                self._ackermann_control = carla.VehicleAckermannControl()
                        elif event.key == K_m:
                            self._control.manual_gear_shift = not self._control.manual_gear_shift
                            self._control.gear = world.player.get_control().gear
                            world.hud.notification('%s Transmission' % ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                        elif self._control.manual_gear_shift and event.key == K_COMMA:
                            self._control.gear = max(-1, self._control.gear - 1)
                        elif self._control.manual_gear_shift and event.key == K_PERIOD:
                            self._control.gear = self._control.gear + 1
                        elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                            if not self._autopilot_enabled and not sync_mode:
                                print("WARNING: asynchronous mode, might cause issues with traffic simulation")
                            self._autopilot_enabled = not self._autopilot_enabled
                            world.player.set_autopilot(self._autopilot_enabled)
                            world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                        elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                            current_lights ^= carla.VehicleLightState.Special1
                        elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                            current_lights ^= carla.VehicleLightState.HighBeam
                        elif event.key == K_l:
                            if not self._lights & carla.VehicleLightState.Position:
                                world.hud.notification("Position lights")
                                current_lights |= carla.VehicleLightState.Position
                            else:
                                world.hud.notification("Low beam lights")
                                current_lights |= carla.VehicleLightState.LowBeam
                            if self._lights & carla.VehicleLightState.LowBeam:
                                world.hud.notification("Fog lights")
                                current_lights |= carla.VehicleLightState.Fog
                            if self._lights & carla.VehicleLightState.Fog:
                                world.hud.notification("Lights off")
                                current_lights ^= carla.VehicleLightState.Position
                                current_lights ^= carla.VehicleLightState.LowBeam
                                current_lights ^= carla.VehicleLightState.Fog
                        elif event.key == K_i:
                            current_lights ^= carla.VehicleLightState.Interior
                            
                        elif event.key == K_z:
                            current_lights &= ~carla.VehicleLightState.RightBlinker
                            current_lights ^= carla.VehicleLightState.LeftBlinker
                            if (current_lights & carla.VehicleLightState.LeftBlinker):
                                channel_blink.play(blink_sound,loops=-1)
                                global last_left_blinker_on_time
                                last_left_blinker_on_time = time.time()
                            else:
                                channel_blink.stop()
                                
                        elif event.key == K_x:
                            current_lights &= ~carla.VehicleLightState.LeftBlinker
                            current_lights ^= carla.VehicleLightState.RightBlinker
                            if (current_lights & carla.VehicleLightState.RightBlinker):
                                channel_blink.play(blink_sound,loops=-1)
                                global last_right_blinker_on_time
                                last_right_blinker_on_time = time.time()
                            else:
                                channel_blink.stop()

                elif event.type == pygame.KEYDOWN:
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        if event.key == K_LEFT:
                            world.hud.switch_direction(0)
                        elif event.key == K_RIGHT:
                            world.hud.switch_direction(1)
                        elif event.key == K_UP:
                            world.hud.switch_direction(2)
                        elif event.key == K_DOWN:
                            world.hud.switch_direction(3) 

            if not self._autopilot_enabled:
                if isinstance(self._control, carla.VehicleControl):
                    self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                    self._control.reverse = self._control.gear < 0
                    if self._control.brake:
                        current_lights |= carla.VehicleLightState.Brake
                    else:
                        current_lights &= ~carla.VehicleLightState.Brake
                    if self._control.reverse:
                        current_lights |= carla.VehicleLightState.Reverse
                    else:
                        current_lights &= ~carla.VehicleLightState.Reverse
                    if current_lights != self._lights:
                        self._lights = current_lights
                        world.player.set_light_state(carla.VehicleLightState(self._lights))
                    if not self._ackermann_enabled:
                        world.player.apply_control(self._control)
                    else:
                        world.player.apply_ackermann_control(self._ackermann_control)
                        self._control = world.player.get_control()
                        world.hud.update_ackermann_control(self._ackermann_control)
                elif isinstance(self._control, carla.WalkerControl):
                    self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                    world.player.apply_control(self._control)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                elif event.type == pygame.KEYUP:
                    if self._is_quit_shortcut(event.key):
                        return True
                    elif event.key == K_BACKSPACE:
                        if self._autopilot_enabled:
                            world.player.set_autopilot(False)
                            world.restart()
                            world.player.set_autopilot(True)
                        else:
                            world.restart()
                    elif event.key == K_v:
                        if image_number<5:
                            image_number+=1
                            channel_flip.play(flip)
                            channel_flip.set_volume(0.5)
                    elif event.key == K_c:
                        if image_number>0 and image_number<5:
                            image_number-=1
                            channel_flip_reverse.play(flip_reverse)
                            channel_flip_reverse.set_volume(0.5)
                        elif image_number>=5:
                            start_stop=1
                            channel_start_engine.play(start_engine)
                            channel_start_engine.set_volume(0.8)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle*0.85 + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            self._control.throttle = 0.3

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[pygame.K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self._control.speed = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[pygame.K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == pygame.K_ESCAPE) or (key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL)


# ========================================================================
# ===============  HUD  ================================================
# ========================================================================
class HUD(object):
    def __init__(self, width, height):
        global end_count
        global total_time
        global red_light
        global wrong_way_time
        global wrong_blinker
        global collision_times
        global Sudden_brake
        global highest_speed
        global cross_line_times
        global Honking_the_horn
        global speedly_time
        global grade
        self.dim = (width, height)
        self._show_info = False
        self.font1 =pygame.font.SysFont(None, 26,bold=True) 
        self.font2 = pygame.font.SysFont(None, 24,bold=True) 
        self.countdown_start_time = time.time()
        self.center_number = 0
        self.remain=300
        self.center_color = (0, 0, 255)
        self.directions = ["left", "right", "straight", "uturn"]
        self.next_directions = ["left", "right", "straight", "uturn"]
        self.current_direction_index = 0
        self.next_direction_index = 0
        self.show_arrow = True
        self.show_next_arrow = True
        self.countdown_last_time=0
        self.image_1 = pygame.image.load("image_1.png")
        self.image_1 = pygame.transform.scale(self.image_1, (720+144*3, 405+81*3))
        self.image_2 = pygame.image.load("image_2.png")
        self.image_2 = pygame.transform.scale(self.image_2, (720+144*3, 405+81*3))
        self.image_3_1 = pygame.image.load("image_3_1.png")
        self.image_3_1 = pygame.transform.scale(self.image_3_1, (720+144*3, 405+81*3))
        self.image_4_1 = pygame.image.load("image_4_1.png")
        self.image_4_1 = pygame.transform.scale(self.image_4_1, (720+144*3, 405+81*3))
        self.image_5 = pygame.image.load("image_5.png")
        self.image_5 = pygame.transform.scale(self.image_5, (720+144*3, 405+81*3))
        self.image_6 = pygame.image.load("end.png")
        self.image_6 = pygame.transform.scale(self.image_6, (720 + 144 * 3, 405 + 81 * 3))
        self.font3 = pygame.font.SysFont("Arial", 24)
        self.font4 = pygame.font.SysFont("PressStart2P-Regular.ttf", 150)
        self.font_color = (255, 255, 255)
    def tick(self, world, clock):
        pass

    def render(self, display):
        pass
        
    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        print(f"[HUD Notif] {text}")

    def error(self, text):
        pass

    def on_world_tick(self, timestamp):
        pass
    #導航箭頭
    def _draw_arrow(self, surface, direction):
        """Navigation system guidance"""
        if not self.show_arrow:
            return
        color = (0, 0, 255)
        arrow_width = 48
        arrow_height = 48
        base_x = 35
        base_y = 80

        if direction == "left":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width /6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6 - arrow_height/4, base_y - arrow_height / 2+ arrow_height / 12- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height / 12- arrow_height / 8- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
                (base_x ,base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
            ]
        elif direction == "right":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width /6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6, base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6, base_y - arrow_height /3+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6 + arrow_width / 4, base_y - arrow_height / 2+ arrow_height / 12- arrow_height / 8+ arrow_height /12),
                (base_x + arrow_width / 6 + arrow_width / 6,base_y - arrow_height / 2- arrow_height / 8- arrow_height / 8+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6,base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x,base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
            ]
        elif direction == "straight":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width / 6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height / 2+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 8, base_y - arrow_height / 2+ arrow_height / 12),
                (base_x + arrow_width / 12, base_y - arrow_height / 2-arrow_height/4+ arrow_height / 12),
                (base_x - arrow_width / 8, base_y - arrow_height / 2+ arrow_height / 12),
                (base_x, base_y - arrow_height / 2+ arrow_height / 12),
            ]
        elif direction == "uturn":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width /6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 3, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 3, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 12),
                (base_x - arrow_width / 3- arrow_width / 8, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height /12),
                (base_x - arrow_width / 4, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 4+ arrow_height / 12),
                (base_x - arrow_width /24, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 12),
                (base_x - arrow_width /6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 12),
                (base_x - arrow_width /6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 12),
                (base_x, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 12),
            ]
        else:
            arrow_points = []

        if arrow_points:
            pygame.draw.polygon(surface, color, arrow_points)
    def _draw_next_arrow(self, surface, direction):
        """Navigation system guidance"""
        if not self.show_next_arrow:
            return
        color = (0, 0, 255)
        arrow_width = 24
        arrow_height = 24
        base_x = 68
        base_y = 83

        if direction == "left":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width /6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6 - arrow_height/4, base_y - arrow_height / 2+ arrow_height / 12- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height / 12- arrow_height / 8- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 6, base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
                (base_x ,base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
            ]
        elif direction == "right":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width /6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6, base_y - arrow_height /3- arrow_height / 8+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6, base_y - arrow_height /3+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6 + arrow_width / 4, base_y - arrow_height / 2+ arrow_height / 12- arrow_height / 8+ arrow_height /12),
                (base_x + arrow_width / 6 + arrow_width / 6,base_y - arrow_height / 2- arrow_height / 8- arrow_height / 8+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 6,base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x,base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
            ]
        elif direction == "straight":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width / 6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height / 2+ arrow_height / 12),
                (base_x + arrow_width / 6 + arrow_width / 8, base_y - arrow_height / 2+ arrow_height / 12),
                (base_x + arrow_width / 12, base_y - arrow_height / 2-arrow_height/4+ arrow_height / 12),
                (base_x - arrow_width / 8, base_y - arrow_height / 2+ arrow_height / 12),
                (base_x, base_y - arrow_height / 2+ arrow_height / 12),
            ]
        elif direction == "uturn":
            arrow_points = [
                (base_x, base_y),
                (base_x + arrow_width /6, base_y),
                (base_x + arrow_width / 6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 3, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 12),
                (base_x - arrow_width / 3, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 12),
                (base_x - arrow_width / 3- arrow_width / 8, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height /12),
                (base_x - arrow_width / 4, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 4+ arrow_height / 12),
                (base_x - arrow_width /24, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 12),
                (base_x - arrow_width /6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 6+ arrow_height / 12),
                (base_x - arrow_width /6, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 12),
                (base_x, base_y - arrow_height / 2- arrow_height / 8+ arrow_height / 6+ arrow_height / 12),
            ]
        else:
            arrow_points = []

        if arrow_points:
            pygame.draw.polygon(surface, color, arrow_points)
    def illustratey(self,display):
        global image_number
        if image_number==0:
            display.blit(self.image_1, (100, 50))
        elif image_number==1:
            display.blit(self.image_2, (100, 50))
        elif image_number==2:
            display.blit(self.image_3_1, (100, 50))
        elif image_number==3:
            display.blit(self.image_4_1, (100, 50))
        elif image_number==4:
            display.blit(self.image_5, (100, 50))
               
    def end_image(self, display):
        global total_time
        global red_light
        global wrong_way_time
        global wrong_blinker
        global collision_times
        global Sudden_brake
        global highest_speed
        global cross_line_times
        global Honking_the_horn
        global speedly_time
        global grade
        self.left_values = [total_time, red_light, wrong_way_time, wrong_blinker, collision_times]
        self.right_values = [highest_speed,cross_line_times,Honking_the_horn, Sudden_brake, speedly_time]
        display.blit(self.image_6, (100, 50))
        # 圖片在畫面上的偏移位置
        offset_x, offset_y = 100, 50

        # 左右欄的數值置中顯示位置
        left_x = offset_x + 540
        right_x = offset_x + 990
        start_y = offset_y + 210
        line_height = 52
        pygame.display.flip()
        time.sleep(1)
        for i, value in enumerate(self.left_values):
            channel_end_sound.play(end_sound)
            y = start_y + i * line_height
            text_surface = self.font3.render(str(value), True, self.font_color)
            text_rect = text_surface.get_rect(center=(left_x, y))
            display.blit(text_surface, text_rect)
            pygame.display.flip()
            time.sleep(1)

        # 畫右欄數值
        for i, value in enumerate(self.right_values):
            channel_end_sound.play(end_sound)
            y = start_y + i * line_height
            text_surface = self.font3.render(str(value), True, self.font_color)
            text_rect = text_surface.get_rect(center=(right_x, y))
            display.blit(text_surface, text_rect)
            pygame.display.flip()
            time.sleep(1)

        # 畫右上角評分
        pygame.display.flip()  
        grade=int(grade)
        grade=f"{grade} "
        score_surface = self.font4.render(str(grade), True, self.font_color)
        score_rect = score_surface.get_rect(topright=(offset_x + 1100, offset_y + 30))
        display.blit(score_surface, score_rect)  
        channel_grade.play(grade__sound)
        pygame.display.flip()
        time.sleep(1)
          
    def render_custom_overlay(self, display):
        """Navigation system screen"""
        global total_time
        overlay_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
        overlay_surf.fill((0, 0, 0,0))
        if  time.time()-self.countdown_last_time>=1:   
            self.remain -=1
            self.countdown_last_time=time.time()
            total_time+=1
        center_str = f"{self.center_number} km/h"
        center_surface = self.font2.render(center_str, True, self.center_color)
        overlay_surf.blit(center_surface, (12,20))
        self._draw_arrow(overlay_surf, self.directions[self.current_direction_index])
        self._draw_next_arrow(overlay_surf, self.next_directions[self.next_direction_index])
        display.blit(overlay_surf, (500,245))
        self.render_distance_display(display, 20,self.remain)
        
    def render_custom_overlay_start(self, display):
        """Navigation system screen"""
        self.countdown_last_time=time.time()
        overlay_surf = pygame.Surface((100, 100), pygame.SRCALPHA)
        overlay_surf.fill((0, 0, 0,0)) 
        center_str = f"{self.center_number} km/h"
        center_surface = self.font2.render(center_str, True, self.center_color)
        overlay_surf.blit(center_surface, (12,20))
        self._draw_arrow(overlay_surf, self.directions[self.current_direction_index])
        self._draw_next_arrow(overlay_surf, self.next_directions[self.next_direction_index])
        display.blit(overlay_surf, (500,245))
        self.render_distance_display(display, 20,self.remain)

    def switch_direction(self, idx):
        """Switch Guidance"""
        if 0 <= idx < len(self.directions):
            self.current_direction_index = idx
            
    def switch_next_direction(self, idx):
        """Switch Guidance"""
        if 0 <= idx < len(self.directions):
            self.next_direction_index = idx
            
    def render_distance_display(self, display, distance_km, remaining_seconds):
        overlay_surf = pygame.Surface((180, 80), pygame.SRCALPHA)
        overlay_surf.fill((0, 0, 0, 180))
        pygame.draw.rect(overlay_surf, (255, 255, 255), overlay_surf.get_rect(), 2)
        font = pygame.font.SysFont('Arial', 20, bold=True)
        m = remaining_seconds // 60
        s = remaining_seconds % 60
        time_text = f"Time left: {m:02d}:{s:02d}"
        time_surface = font.render(time_text, True, (255, 255, 0))
        overlay_surf.blit(time_surface, (20, 30))

        screen_width, screen_height = display.get_size()
        display.blit(overlay_surf, (screen_width - 290, 40))

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255,255,255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0,0,0,0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0*self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines)*self.line_space + 12)
        self.pos = (0.5*width - 0.5*self.dim[0], 0.5*height - 0.5*self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0,0,0,0))
        for n,line in enumerate(lines):
            text_texture = self.font.render(line, True, (255,255,255))
            self.surface.blit(text_texture, (22, n*self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.collision_detected = False
        self.collision_object = None
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda e: CollisionSensor._on_collision(weak_self, e))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame,intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision_detected = True
        self.collision_object = get_actor_display_name(event.other_actor)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        self.line_crossed = False
        self.crossed_lines = []
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda e: LaneInvasionSensor._on_invasion(weak_self, e))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.line_crossed = True
        lane_types = set(x.type for x in event.crossed_lane_markings)
        self.crossed_lines=lane_types


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0,z=2.8)), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda e: GnssSensor._on_gnss_event(weak_self, e))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self=weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0,0,0)
        self.gyroscope = (0,0,0)
        self.compass = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self=weakref.ref(self)
        self.sensor.listen(lambda data:IMUSensor._IMU_callback(weak_self,data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self=weak_self()
        if not self:
            return
        limits=(-99.9,99.9)
        self.accelerometer = (
            max(limits[0],min(limits[1],sensor_data.accelerometer.x)),
            max(limits[0],min(limits[1],sensor_data.accelerometer.y)),
            max(limits[0],min(limits[1],sensor_data.accelerometer.z))
        )
        self.gyroscope=(
            max(limits[0],min(limits[1],math.degrees(sensor_data.gyroscope.x))),
            max(limits[0],min(limits[1],math.degrees(sensor_data.gyroscope.y))),
            max(limits[0],min(limits[1],math.degrees(sensor_data.gyroscope.z)))
        )
        self.compass=math.degrees(sensor_data.compass)


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor=None
        self._parent=parent_actor
        bound_x=0.5+self._parent.bounding_box.extent.x
        bound_y=0.5+self._parent.bounding_box.extent.y
        bound_z=0.5+self._parent.bounding_box.extent.z
        self.velocity_range=7.5
        world=self._parent.get_world()
        self.debug=world.debug
        bp=world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov','35')
        bp.set_attribute('vertical_fov','20')
        self.sensor=world.spawn_actor(
            bp,carla.Transform(
                carla.Location(x=bound_x+0.05,z=bound_z+0.05),
                carla.Rotation(pitch=5)
            ),attach_to=self._parent
        )
        weak_self=weakref.ref(self)
        self.sensor.listen(lambda data:RadarSensor._Radar_callback(weak_self,data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self=weak_self()
        if not self:
            return
        current_rot=radar_data.transform.rotation
        for detect in radar_data:
            azi=math.degrees(detect.azimuth)
            alt=math.degrees(detect.altitude)
            fw_vec=carla.Vector3D(x=detect.depth-0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch+alt,
                    yaw=current_rot.yaw+azi,
                    roll=current_rot.roll
                )
            ).transform(fw_vec)
            def clamp(min_v,max_v,value):
                return max(min_v,min(value,max_v))
            norm_velocity = detect.velocity/self.velocity_range
            r=int(clamp(0.0,1.0,1.0-norm_velocity)*255.0)
            g=int(clamp(0.0,1.0,1.0-abs(norm_velocity))*255.0)
            b=int(abs(clamp(-1.0,0.0,-1.0-norm_velocity))*255.0)
            self.debug.draw_point(
                radar_data.transform.location+fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r,g,b)
            )
# ==============================================================================
# -- TrafficManager --------------------------------------------------------------------
# ==============================================================================
"""Autonomous vehicle control"""
class TrafficManager:
    def __init__(self, carla_world, client, num_vehicles=0):
        self.carla_world = carla_world 
        self.client = client   
        self.vehicles = []
        self.num_vehicles = num_vehicles
        self.traffic_manager = self.client.get_trafficmanager() 
        self._spawn_vehicles()

    def _spawn_vehicles(self):
        spawn_points = self.carla_world.get_map().get_spawn_points()
        blueprints = self.carla_world.get_blueprint_library().filter('vehicle.audi.A2')
        num_spawn_points = len(spawn_points)
        num_vehicles_to_spawn = min(self.num_vehicles, num_spawn_points)

        for _ in range(num_vehicles_to_spawn):
            blueprint = random.choice(blueprints)
            spawn_point = random.choice(spawn_points)
            vehicle = self.carla_world.try_spawn_actor(blueprint, spawn_point)

            if vehicle:
                vehicle.set_autopilot(True, self.traffic_manager.get_port()) 
                self.vehicles.append(vehicle)

    def destroy_vehicles(self):
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.vehicles = []

# ========================================================================
# ===============  CameraManager  ========================================
# ========================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self._parent = parent_actor
        self.hud = hud
        self.sensor = None
        self.surface = None
        self.recording = False

        self._camera_transforms = [
            carla.Transform(carla.Location(x=-0.1, y=-0.35, z=1.35), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))
        ]
        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
        ]
        bp_lib = self._parent.get_world().get_blueprint_library()
        for item in self.sensors:
            bp = bp_lib.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self.index = None

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (self.sensors[index][0] != self.sensors[self.index][0])
        if needs_respawn:
            if self.sensor:
                self.sensor.destroy()
                self.surface = None
            bp = self.sensors[index][-1]
            self.sensor = self._parent.get_world().spawn_actor(
                bp, self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_camera(self):
        pass

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def update_lane_and_turn_state(world, dt, current_time):
    """Turn and lane change detection"""
    global previous_lane_id, turning, turn_side, turn_initiation_heading, previous_heading
    global last_left_blinker_on_time, last_right_blinker_on_time

    lanechange_left = 0
    lanechange_right = 0
    left_turn = 0
    right_turn = 0
    correct_blinker_for_lanechange = 0
    correct_blinker_for_turn = 0

    player = world.player
    if not player:
        return lanechange_left, lanechange_right, left_turn, right_turn, correct_blinker_for_lanechange, correct_blinker_for_turn

    transform = player.get_transform()
    heading = transform.rotation.yaw
    heading = (heading + 180) % 360 - 180

    map_ = world.world.get_map()
    waypoint = map_.get_waypoint(transform.location, project_to_road=True)
    current_lane_id = waypoint.lane_id

    control = player.get_control()
    steer = control.steer

    if previous_lane_id is None:
        previous_lane_id = current_lane_id
    else:
        if current_lane_id != previous_lane_id:
            if steer < 0:
                lanechange_left = 1
                if (current_time - last_left_blinker_on_time) >= 1.0:
                    correct_blinker_for_lanechange = 1
            else:
                lanechange_right = 1
                if (current_time - last_right_blinker_on_time) >= 1.0:
                    correct_blinker_for_lanechange = 1
            previous_lane_id = current_lane_id

    global turning, turn_side
    if not turning:
        if abs(steer) >= steer_threshold:
            turning = True
            turn_side = 'left' if steer < 0 else 'right'
            turn_initiation_heading = heading
    else:
        if abs(steer) < 0.05:
            heading_diff = abs(heading - turn_initiation_heading)
            if heading_diff > 70:
                if turn_side == 'left':
                    left_turn = 1
                    if (current_time - last_left_blinker_on_time) >= 1.0:
                        correct_blinker_for_turn = 1
                else:
                    right_turn = 1
                    if (current_time - last_right_blinker_on_time) >= 1.0:
                        correct_blinker_for_turn = 1
            turning = False
            turn_side = None

    return (lanechange_left, lanechange_right,
            left_turn, right_turn,
            correct_blinker_for_lanechange, correct_blinker_for_turn)
    
def control_vehicle3(world, state):
    control_2 = carla.VehicleControl()
    # throttle 與 steer 值資料表
    throttle_steer_pairs = [
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.3, -0.5), (0.3, -0.5),
    ]
    if state==23:
        channel_slow_moving.play(slow_moving)
        channel_slow_moving.set_volume(0.8) 
    if state==44:
        channel_car_horn_beep.play(car_horn_beep)
        channel_car_horn_beep.set_volume(1)
    if 1 <= state <= len(throttle_steer_pairs):
        raw_throttle, steer = throttle_steer_pairs[state - 1]
        control_2.throttle = raw_throttle
        control_2.steer = steer
    else:
        world.vehicle3.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)) 
        return 1
    if world.vehicle3 is not None:
        world.vehicle3.apply_control(control_2)
    return 0

def control_vehicle5(world, state):
    control_2 = carla.VehicleControl()
    # throttle 與 steer 值資料表
    throttle_steer_pairs = [
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    ]

    if 1 <= state <= len(throttle_steer_pairs):
        raw_throttle, steer = throttle_steer_pairs[state - 1]
        control_2.throttle = raw_throttle
        control_2.steer = steer
        light_state = carla.VehicleLightState.Position | carla.VehicleLightState.HighBeam
        world.vehicle5.set_light_state(carla.VehicleLightState(light_state)) 
    else:
        world.vehicle5.set_light_state(carla.VehicleLightState.NONE)
        world.vehicle5.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)) 
        world.vehicle5.set_location(carla.Location(x=175.00, y=-33.667, z=0.2)) 
        return 1
    if world.vehicle5 is not None:
        world.vehicle5.apply_control(control_2)
    return 0

def control_vehicle7(world, state):
    control_2 = carla.VehicleControl()
    # throttle 與 steer 值資料表
    throttle_steer_pairs = [
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5), (0.6, -0.5),
    (0.6, -0.5),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.5, 0.0),
    (0.6, 0.2), (0.6, 0.2),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.5), (0.6, 0.5),
    (0.6, 0.4), (0.6, 0.4),
    (0.6, 0.4), (0.6, 0.4),
    ]
    if 1 <= state <= len(throttle_steer_pairs):
        raw_throttle, steer = throttle_steer_pairs[state - 1]
        control_2.throttle = raw_throttle
        control_2.steer = steer
        light_state = carla.VehicleLightState.Position | carla.VehicleLightState.HighBeam
        world.vehicle7.set_light_state(carla.VehicleLightState(light_state)) 
    else:
        world.vehicle7.set_light_state(carla.VehicleLightState.NONE)
        world.vehicle7.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)) 
        world.vehicle7.set_location(carla.Location(x=165.00, y=-33.667, z=0.2))
        return 1
    if world.vehicle7 is not None:
        world.vehicle7.apply_control(control_2)
    return 0

def control_vehicle9(world, state):
    control_2 = carla.VehicleControl()
    # throttle 與 steer 值資料表
    throttle_steer_pairs = [
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.0 ,0.0),(0.0 ,0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.6, 0.0), (0.6, 0.0),
    (0.2, -0.1), (0.2, -0.1),
    (0.2, -0.1), (0.2, -0.1),
    (0.2, -0.1), (0.2, -0.1),
    (0.2, 0.4), (0.2, 0.4),
    (0.2, 0.4), (0.2, 0.4),
    (0.2, 0.4), (0.2, 0.4),
    (0.1, 0.4), (0.1, 0.4),
    (0.1, 0.4), (0.1, 0.4),
    (0.2, -0.1), (0.2, -0.1),
    (0.2, -0.1), (0.2, -0.1),
    (0.2, -0.2), (0.2, -0.2),
    (0.2, -0.2), (0.2, -0.2),
    (0.2, -0.2), (0.2, -0.2),
    (0.2, -0.2), (0.2, -0.2),
    (0.2, -0.2), (0.2, -0.2),
    (0.2, -0.2), (0.2, -0.2),
    (0.2, -0.2), (0.2, -0.2),
    ] 
    if state==15:
        channel_slow_moving.play(slow_moving)
        channel_slow_moving.set_volume(0.8) 
    if 1 <= state <= len(throttle_steer_pairs):
        raw_throttle, steer = throttle_steer_pairs[state - 1]
        control_2.throttle = raw_throttle
        control_2.steer = steer
    else:
        world.vehicle9.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)) 
         
        return 1
    if world.vehicle9 is not None:
        world.vehicle9.apply_control(control_2)
    return 0

def control_low_case_car(world, state):
    control_2 = carla.VehicleControl()

    if 1 <= state <= 280:
        control_2.throttle = 0
        control_2.steer = 0
        light_state = carla.VehicleLightState.Position | carla.VehicleLightState.HighBeam
        world.vehicle4.set_light_state(carla.VehicleLightState(light_state))   
    else:
        world.vehicle4.set_light_state(carla.VehicleLightState.NONE)
        channel_start_stop.stop()
        return 1
    if world.vehicle4 is not None:
        world.vehicle4.apply_control(control_2)
    return 0
volume_states = {
    "driving": {
        "channel": channel_driving,
        "current": 0.0,
        "start_volume": 0.0,
        "target": 0.0,
        "start_time": time.time(),
        "duration": 0.5
    },
    "iddle": {
        "channel": channel_iddle,
        "current": 0.1,
        "start_volume": 0.1,
        "target": 0.1,
        "start_time": time.time(),
        "duration": 0.5
    },
    "brake": {
        "channel": channel_brake,
        "current": 0.0,
        "start_volume": 0.0,
        "target": 0.0,
        "start_time": time.time(),
        "duration": 0.5
    }
}
def set_target_volume(name, target_volume, duration=0.3):
    state = volume_states[name]
    target_volume = max(0.0, min(1.0, target_volume))
    if abs(state["target"] - target_volume) < 0.01:
        return  # 幾乎沒變化就不用設
    state["start_time"] = time.time()
    state["start_volume"] = state["current"]
    state["target"] = target_volume
    state["duration"] = duration
def update_volumes():
    now = time.time()
    for state in volume_states.values():
        elapsed = now - state["start_time"]
        t = min(1.0, elapsed / state["duration"])  # 正規化進度 0~1
        new_volume = (1 - t) * state["start_volume"] + t * state["target"]
        state["current"] = new_volume
        state["channel"].set_volume(new_volume)
def game_loop(args):
    """
    Main Operations
    It will record the events that occur during the process, write them into Excel, and also record the video simultaneously.
    """
    pygame.init()
    pygame.font.init()
    
    world = None
    writer = imageio.get_writer('output1-2.mp4', fps=50, quality=1, codec='libx264', bitrate='2000k')

    excel_path = "output_data1-2.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Simulation Data"
    sheet.append([
        "Timestamp", "Speed (km/h)", "speeding", "SpeedDiff", "acceleration",
        "throttle", "brake", "steer", "reverse", "wrong_way",
        "Front_Vehicle", "Front_Vehicle_ID", "Front_Distance",
        "Lane_Offset", "Collision", "Collision_Object",
        "Cross_Line", "Crossed_Lines",
        "Left_Turn", "Right_Turn",
        "LaneChange_Left", "LaneChange_Right",
        "Left_Blinker", "Right_Blinker",
        "Correct_LaneChange_Blinker",
        "Correct_Turn_Blinker",
        "x","y","z"
    ])
    workbook.save(excel_path)
    global image_number
    global start_stop
    global total_time
    global wrong_way_time
    global wrong_blinker
    global collision_times
    global Sudden_brake
    global highest_speed
    global cross_line_times
    global Honking_the_horn
    global speedly_time
    global grade
    stop_wrong_way=0
    wrong_way=0
    Sudden_brake_time=time.time()
    collision_time=time.time()
    cross_line_time=time.time()
    pass_level=0
    print_time = 0
    speed_limit = 80
    last_time = time.time()
    old_speed = 0
    cycle = 0.1
    take_place_event=0
    take_place_event_3=0
    take_place_event_2=0
    state=0
    state_2=0
    state_3=0
    #設定事件觸發點
    event_location1=carla.Location(x=63.247, y=-23.263, z=0)
    event_location2=carla.Location(x=24.521, y=-10.644, z=0)
    event_location3=carla.Location(x=24.461, y=15.666, z=0)
    event_location4=carla.Location(x=41.948, y=32.518, z=0)
    event_location5=carla.Location(x=156.71, y=32.768, z=0)
    event_location6=carla.Location(x=166.88, y=50.698, z=0)
    event_location7=carla.Location(x=168.05, y=-18.07, z=0)
    event_location8=carla.Location(x=196.982, y=38.454, z=0)
    event_location9=carla.Location(x=81.98, y=32.768, z=0)
    event_location10=carla.Location(x=124.33, y=32.53, z=0)
    current_location_state=0
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF )
        pygame.mouse.set_visible(False)

        hud = HUD(args.width, args.height)
        carla_world = client.get_world()
        carla_world.set_weather(carla.WeatherParameters.ClearNight)
        world = World(carla_world, hud, args)

        controller = KeyboardControl(world, args.autopilot)
        clock = pygame.time.Clock()
        world.hud.switch_direction(2)
        world.hud.switch_next_direction(0)
        
        while True:    
            clock.tick_busy_loop(50)
            fps = clock.get_fps()
            pygame.display.set_caption(f"CARLA Simulation - FPS: {fps:.2f}")
            world.tick(clock)
            world.render(display)
            current_time = time.time()
            if controller.parse_events(client, world, clock, args.sync):
                    break
            if image_number<=4:
                world.hud.illustratey(display)
                pygame.display.flip()
            else:
                if start_stop==1:
                    world.hud.render_custom_overlay(display)
                    pygame.display.flip()
                    if (current_time - last_time) >= cycle:
                        last_time = current_time
                        print_time += cycle
                        player = world.player
                        if player:
                            velocity = player.get_velocity()
                            control = player.get_control()
                            transform = player.get_transform()
                            rotation = transform.rotation
                            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)*1.5    
                            if speed >highest_speed:
                                highest_speed=int(speed)
                            hud.center_number = int(speed)
                            speeding = "1" if speed > speed_limit else "0"
                            if speed > speed_limit:
                                hud.center_color=(255,0,0)
                                speedly_time+=1
                            else:
                                hud.center_color=(0, 0, 255)
                            speed_diff = speed - old_speed
                            old_speed = speed
                            acceleration = speed_diff / cycle
                            throttle = control.throttle
                            steer = control.steer
                            brake = control.brake
                            reverse = control.reverse
                            if acceleration<=-30 and brake==1 and time.time()-Sudden_brake_time>=1 and reverse=="FALSE":
                                Sudden_brake+=1
                                Sudden_brake_time=time.time()
                            
                            if stop_wrong_way==0:
                                if world.is_driving_against_traffic():
                                    wrong_way = "1" if world.is_driving_against_traffic() else "0"
                                    wrong_way_time+=1

                            front_v, dist = world.get_front_vehicle_distance()
                            if front_v:
                                front_v_id = front_v.id
                                fv_flag = "1"
                            else:
                                front_v_id = "None"
                                dist = "None"
                                fv_flag = "0"

                            lane_offset = world.get_lane_offset()

                            collision = "0"
                            collision_obj = "None"
                            if world.collision_sensor.collision_detected:
                                collision = "1"
                                collision_obj = world.collision_sensor.collision_object
                                world.collision_sensor.collision_detected = False
                                world.collision_sensor.collision_object = None
                                if time.time()-collision_time>=1:
                                    collision_times+=1
                                    collision_time=time.time()

                            cross_line = "0"
                            crossed_lines = "None"
                            if world.lane_invasion_sensor.line_crossed:
                                cross_line = "1"
                                line_set = world.lane_invasion_sensor.crossed_lines
                                if line_set:
                                    crossed_lines = ','.join([str(x).split('.')[-1] for x in line_set])
                                if crossed_lines=="SolidSolid" and time.time()-cross_line_time>=1:
                                   cross_line_times+=1 
                                   cross_line_time=time.time()
                                world.lane_invasion_sensor.line_crossed = False
                                world.lane_invasion_sensor.crossed_lines = []

                            (lc_left, lc_right,
                                turn_left, turn_right,
                                correct_lane_blinker,
                                correct_turn_blinker
                            ) = update_lane_and_turn_state(world, cycle, current_time)

                            lights = player.get_light_state()
                            left_blinker_on = int(bool(lights & carla.VehicleLightState.LeftBlinker))
                            right_blinker_on = int(bool(lights & carla.VehicleLightState.RightBlinker))
                                
                            current_location = player.get_location()
                    
                            if current_location_state==0:
                                dx = current_location.x - event_location1.x
                                dy = current_location.y - event_location1.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(0)
                                    world.hud.switch_next_direction(2)
                                    current_location_state+=1
                                    
                            elif current_location_state==1:
                                dx = current_location.x - event_location2.x
                                dy = current_location.y - event_location2.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(2)
                                    world.hud.switch_next_direction(0)
                                    if left_blinker_on!=1:
                                        wrong_blinker+=1
                                    current_location_state+=1
                                    
                            elif current_location_state==2:
                                dx = current_location.x - event_location3.x
                                dy = current_location.y - event_location3.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(0)
                                    world.hud.switch_next_direction(2)
                                    current_location_state+=1
                                    
                            elif current_location_state==3:
                                dx = current_location.x - event_location4.x
                                dy = current_location.y - event_location4.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(2)
                                    world.hud.switch_next_direction(2)
                                    if left_blinker_on!=1:
                                        wrong_blinker+=1
                                    current_location_state+=1
                                    
                            elif current_location_state==4:
                                dx = current_location.x - event_location9.x
                                dy = current_location.y - event_location9.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(2)
                                    world.hud.switch_next_direction(2)
                                    current_location_state+=1
                                    
                            elif current_location_state==5:
                                dx = current_location.x - event_location10.x
                                dy = current_location.y - event_location10.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(2)
                                    hud.show_next_arrow = False
                                    current_location_state+=1  
                                        
                            elif current_location_state==6:
                                dx = current_location.x - event_location5.x
                                dy = current_location.y - event_location5.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5:
                                    world.hud.switch_direction(2)
                                    hud.show_arrow = False
                                    current_location_state+=1
                                    
                            elif current_location_state ==7:
                                dx = current_location.x - event_location6.x
                                dy = current_location.y - event_location6.y
                                distance = math.sqrt(dx * dx + dy * dy)
                                if distance <= 5 and -30 <= rotation.yaw <= 30:  
                                    world.vehicle9.set_location(carla.Location(x=196.92, y=30, z=0.2))                      
                                    current_location_state += 1
                                    take_place_event_2=1
                                    channel_car_start.play(car_start_sound)
                                    channel_car_start.set_volume(0.3,0.8)
                                    
                            elif current_location_state==8:
                                dx = current_location.x - event_location7.x
                                dy = current_location.y - event_location7.y  
                                if math.sqrt(dx*dx + dy*dy)<=3.5 and -30 <= rotation.yaw <= 30:
                                    current_location_state+=1
                                    world.vehicle3.set_location(carla.Location(x=174.93, y=-7.344, z=0.2)) 
                                    take_place_event = 1  
                                    channel_car_start.play(car_start_sound)
                                    channel_car_start.set_volume(0.3,0.8)
                                    
                            elif current_location_state==9:
                                world.vehicle6.set_location(carla.Location(x=145.00, y=-33.667, z=0.2))
                                current_location_state+=1
                                
                            elif current_location_state == 10:
                                dx = current_location.x - event_location8.x
                                dy = current_location.y - event_location8.y
                                distance = math.sqrt(dx * dx + dy * dy)
                                if distance <= 3.5 and -129 <= rotation.yaw <= -59:                       
                                    current_location_state += 1
                                    take_place_event_3 = 1  
                                    channel_start_stop.play(start_stop_sound)
                                    channel_start_stop.set_volume(0.8)    

                            elif current_location_state == 11:
                                if 170 <= rotation.yaw <= 190:                       
                                    current_location_state += 1
                                    world.vehicle4.set_location(carla.Location(x=154.93, y=-7.344, z=0.2))      
                            
                            elif current_location_state == 12:
                                if  current_location.x > 190 and 18<current_location.y <21:                       
                                    current_location_state += 1
                                    print("pass")
                                    pass_level=1
                                    
                            if take_place_event == 1:
                                state += 1
                                control_vehicle3(world, state)                     
                                if control_vehicle5(world, state)==1:
                                    take_place_event = 0 
                            
                            if take_place_event_2 == 1:
                                state_2 += 1
                                control_vehicle9(world, state_2)                     
                                if control_vehicle7(world, state_2)==1:
                                    take_place_event_2 = 0 
                                    
                            if take_place_event_3 == 1:
                                state_3 += 1
                                if control_low_case_car(world, state_3) == 1:
                                    take_place_event_3 = 0 
                                    
                                        
                            x=current_location.x
                            y=current_location.y
                            z=current_location.z
                            sheet.append([
                            print_time, speed, speeding, speed_diff, acceleration,
                                throttle, brake, steer, reverse, wrong_way,
                                fv_flag, front_v_id, dist,
                                lane_offset, collision, collision_obj,
                                cross_line, crossed_lines,
                                turn_left, turn_right,
                                lc_left, lc_right,
                                left_blinker_on,
                                right_blinker_on,
                                correct_lane_blinker,
                                correct_turn_blinker,
                                x,y,z
                            ]) 
                    velocity = player.get_velocity()
                    control = player.get_control()
                    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)*1.5                              
                    brake = control.brake
                    if not channel_driving.get_busy():
                        channel_driving.play(driving_sound)
                    if not channel_iddle.get_busy():
                        channel_iddle.play(iddle)
                        channel_iddle.set_volume(0.1)

                    # 根據 speed 設定目標音量
                    if speed <= 5:
                        set_target_volume("driving", 0.0)
                    elif speed <= 10:
                        set_target_volume("driving", 0.3)
                    elif speed <= 20:
                        set_target_volume("driving", 0.3)
                    elif speed <= 30:
                        set_target_volume("driving", 0.4)
                    elif speed <= 40:
                        set_target_volume("driving", 0.5)
                    elif speed <= 50:
                        set_target_volume("driving", 0.6)
                    elif speed <= 60:
                        set_target_volume("driving", 0.7)
                    elif speed <= 70:
                        set_target_volume("driving", 0.8)
                    elif speed <= 80:
                        set_target_volume("driving", 0.9)
                    else:
                        set_target_volume("driving", 1.0)

                    # 煞車狀態處理
                    if brake >= 0.8 and speed >= 10:
                        if not channel_brake.get_busy():
                            channel_brake.play(brake_sound)
                        if speed >= 80:
                            set_target_volume("brake", 1.0)
                        elif speed >= 70:
                            set_target_volume("brake", 0.9)
                        elif speed >= 60:
                            set_target_volume("brake", 0.8)
                        elif speed >= 50:
                            set_target_volume("brake", 0.6)
                        elif speed >= 40:
                            set_target_volume("brake", 0.4)
                        else:
                            set_target_volume("brake", 0.0)
                    else:
                        channel_brake.stop()
                        volume_states["brake"]["current"] = 0.0
                        volume_states["brake"]["target"] = 0.0
                    update_volumes()
                elif start_stop==0 and pass_level==0:
                    if pass_level==0:
                        world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                        world.hud.render_custom_overlay_start(display)
                        pygame.display.flip()
                        if channel_driving.get_busy():
                            channel_driving.stop()
                        if channel_Car_reversing_warning_sound.get_busy():
                            channel_Car_reversing_warning_sound.stop()
                        if channel_iddle.get_busy():
                            channel_iddle.fadeout(1000)
                elif start_stop==0 and pass_level==2:
                    if channel_driving.get_busy():
                        channel_driving.stop()
                    if channel_Car_reversing_warning_sound.get_busy():
                        channel_Car_reversing_warning_sound.stop()
                    if channel_iddle.get_busy():
                        channel_iddle.fadeout(1000)
                    image_number=5
                    world.hud.end_image(display)
                    pass_level+=1
                    
                elif pass_level==1:
                    if total_time>=300:
                        grade-=10
                    if wrong_blinker<=5:
                        grade-=wrong_blinker*2
                    else:
                       grade-=10
                    if collision_times<=5:
                        grade-=collision_times*2
                    else:
                       grade-=10  
                    if Sudden_brake<=5:
                        grade-=Sudden_brake*2
                    else:
                       grade-=10   
                    if cross_line_times<=5:
                        grade-=cross_line_times*2
                    else:
                       grade-=10 
                    if Honking_the_horn<=5:
                        grade-=Honking_the_horn*2
                    else:
                       grade-=10  
                    m = total_time // 60
                    s = total_time % 60
                    total_time=f"{m:02d} m:{s:02d} s"
                    wrong_way_time//=15
                    if wrong_way_time<=100:
                       grade-= wrong_way_time/10
                    else:
                        grade-=10
                    wrong_way_m = wrong_way_time // 60
                    wrong_way_s = wrong_way_time % 60
                    wrong_way_time=f"{wrong_way_m:02d} m:{wrong_way_s:02d} s"
                    highest_speed=f"{highest_speed} km/h"
                    speedly_time//=10
                    if speedly_time<=100:
                       grade-= speedly_time/10
                    else:
                        grade-=10
                    speedly_m = speedly_time // 60
                    speedly_s = speedly_time % 60
                    speedly_time=f"{speedly_m:02d} m:{speedly_s:02d} s"
                    pass_level+=1
                elif pass_level==3:
                    world.player.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            #frame = pygame.surfarray.array3d(display)
            #frame = frame.transpose((1,0,2))
            #writer.append_data(frame)
        try:
            workbook.save(excel_path)
        except Exception as e:
            logging.error("Error saving Excel: %s", str(e))    
        writer.close()
        if world:
            world.destroy()

        workbook.close()
        pygame.quit()
        sys.exit()

    except Exception as e:
        logging.error("Exception: %s", str(e), exc_info=True)
        if world:
            world.destroy()
        writer.close()
        workbook.close()
        pygame.quit()
        sys.exit()

def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-v','--verbose',action='store_true',dest='debug',help='print debug information')
    argparser.add_argument('--host',metavar='H',default='127.0.0.1',help='IP of the host server')
    argparser.add_argument('-p','--port',metavar='P',default=2000,type=int,help='TCP port')
    argparser.add_argument('-a','--autopilot',action='store_true',help='enable autopilot')
    argparser.add_argument('--res',metavar='WIDTHxHEIGHT',default='1280x720',help='window resolution')
    argparser.add_argument('--filter',metavar='PATTERN',default='vehicle.*',help='actor filter')
    argparser.add_argument('--generation',metavar='G',default='2',help='restrict to certain actor generation')
    argparser.add_argument('--rolename',metavar='NAME',default='hero',help='actor role name')
    argparser.add_argument('--gamma',default=2.2,type=float,help='Gamma correction of the camera')
    argparser.add_argument('--sync',action='store_true',help='Activate synchronous mode execution')
    args=argparser.parse_args()

    args.width,args.height=[int(x) for x in args.res.split('x')]

    log_level=logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s',level=log_level)

    logging.info('listening to server %s:%s',args.host,args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()


    

