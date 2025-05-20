import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
import math
import time
from collections import deque
from typing import List, Tuple

# Constants
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60
TRAIN_LENGTH = 3.0  # Longer train for more realistic proportions
TRAIN_WIDTH = 1.2   # Slightly wider
TRAIN_HEIGHT = 0.8  # Taller for better aesthetics
TRACK_LENGTH = 60.0
MAGNET_SPACING = 2.0  # Increased spacing for better performance
FIELD_STRENGTH = 1.5
MAX_SPEED = 40.0
ACCELERATION = 1.0
DAMPING = 0.02
MAGNETIC_FORCE_RANGE = 2.5
USE_DISPLAY_LISTS = True  # Enable display lists for better performance

class ShaderManager:
    """Handles shader compilation and management"""
    @staticmethod
    def create_shader(vertex_source: str, fragment_source: str) -> int:
        """Compile and link shader program"""
        vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
        return compileProgram(vertex_shader, fragment_shader)

class MaglevSimulation:
    def __init__(self):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Advanced Maglev Simulation")
        
        # Simulation state
        self.train_pos = -TRACK_LENGTH/2
        self.train_speed = 0.0
        self.train_acceleration = 0.0
        self.is_running = False
        self.show_field_lines = True
        self.show_field_arrows = True
        self.show_trail = True
        self.field_strength = FIELD_STRENGTH
        self.track_resistance = 0.01
        
        # Camera control - enhanced for free movement
        self.camera_distance = 25.0
        self.camera_angle_x = 30.0
        self.camera_angle_y = -30.0
        self.camera_pos = [0, 8, 0]
        self.camera_target = [0, 0, 0]  # Point camera is looking at
        self.camera_up = [0, 1, 0]      # Up vector
        self.mouse_dragging = False
        self.mouse_last_pos = (0, 0)
        self.camera_mode = "free"       # "orbit" or "free"
        self.third_person_view = True   # Enable third person view
        
        # Visualization data
        self.field_lines = []
        self.field_arrows = []
        self.track_vertices = []
        self.train_history = deque(maxlen=200)
        self.generate_geometry()
        
        # 3D Model loading
        self.train_model = None
        try:
            self.load_train_model()
        except Exception as e:
            print(f"Could not load train model: {e}")
        
        # Shaders
        self.shaders = self.setup_shaders()
        self.current_shader = None
        self.use_shaders = True
        
        # Timing
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # UI state
        self.show_help = True
        self.show_debug = True
        
        self.setup_opengl()
    
    def setup_shaders(self) -> dict:
        """Create all shader programs"""
        vertex_shader = """
#version 120
varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;

void main() {
    v_position = gl_Vertex.xyz;
    v_normal = gl_Normal;
    v_color = gl_Color;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""
        
        magnetic_field_fragment = """
#version 120
varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;

uniform float time;

void main() {
    float pulse = 0.7 + 0.3 * sin(time * 3.0 + v_position.x * 5.0);
    vec3 base_color = v_color.rgb;
    float edge = max(0.0, dot(v_normal, vec3(0.0, 1.0, 0.0)));
    edge = pow(edge, 2.0);
    
    gl_FragColor = vec4(mix(base_color, base_color * 1.5, edge) * pulse, v_color.a);
}
"""
        
        train_fragment = """
#version 120
varying vec3 v_position;
varying vec3 v_normal;
varying vec4 v_color;

uniform float time;

void main() {
    float spec = pow(max(0.0, dot(v_normal, normalize(vec3(0.5, 1.0, 0.5)))), 32.0);
    float pulse = 0.8 + 0.2 * sin(time * 2.0 + v_position.x * 3.0);
    vec3 highlight = vec3(spec * pulse);
    
    gl_FragColor = vec4(v_color.rgb + highlight, v_color.a);
}
"""
        
        return {
            'magnetic_field': ShaderManager.create_shader(vertex_shader, magnetic_field_fragment),
            'train': ShaderManager.create_shader(vertex_shader, train_fragment),
            'default': None
        }
    
    def setup_opengl(self):
        """Configure OpenGL rendering context"""
        glClearColor(0.05, 0.05, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 10, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1, 1, 1, 1])
        glMaterialfv(GL_FRONT, GL_SHININESS, [50])
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
    
    def generate_geometry(self):
        """Generate all track and field visualization geometry"""
        # Track vertices - reduced segment count for better performance
        segments = 150
        self.track_vertices = [(-TRACK_LENGTH/2 + (i/segments)*TRACK_LENGTH, 0, 0) 
                              for i in range(segments + 1)]
        
        # Magnetic field visualization
        num_magnets = int(TRACK_LENGTH / MAGNET_SPACING) + 1
        
        # Adjust line resolution based on performance needs
        line_resolution = 16  # Reduced from 24 for better performance
        
        # Pre-allocate lists for better memory efficiency
        self.field_lines = []
        self.field_arrows = []
        
        # Generate field geometry
        for i in range(num_magnets):
            magnet_x = -TRACK_LENGTH/2 + i * MAGNET_SPACING
            
            # Field lines (circles around magnets) - more efficient generation
            for j in range(line_resolution):
                angle1 = 2 * math.pi * j / line_resolution
                angle2 = 2 * math.pi * (j + 1) / line_resolution
                radius = 0.8
                
                # Calculate positions
                x1 = magnet_x + radius * math.cos(angle1)
                z1 = radius * math.sin(angle1)
                x2 = magnet_x + radius * math.cos(angle2)
                z2 = radius * math.sin(angle2)
                
                # Add to field lines list
                self.field_lines.extend([(x1, 0.1, z1), (x2, 0.1, z2)])
            
            # Field direction arrows - reduced count for better performance
            for direction in [-1, 1]:
                for step in np.linspace(0.3, 1.4, 4):  # Reduced from 6 steps to 4
                    x = magnet_x
                    z = direction * step
                    self.field_arrows.append((x, 0.1, z, 0, 0, direction * 0.3))
        
        # Create display lists for static geometry if enabled
        # if USE_DISPLAY_LISTS:
        #     self.create_display_lists()
    
    def calculate_magnetic_force(self, train_x: float) -> float:
        """Calculate magnetic force acting on the train"""
        closest_magnet = round(train_x / MAGNET_SPACING) * MAGNET_SPACING
        distance = train_x - closest_magnet
        
        # Only affect when close to magnet
        if abs(distance) > MAGNETIC_FORCE_RANGE:
            return 0.0
        
        # Inverse square law with sign based on position
        force = self.field_strength / (distance**2 + 0.1) * np.sign(-distance)
        return force
    
    def update(self):
        """Update simulation physics"""
        current_time = time.time()
        delta_time = min(0.1, current_time - self.last_time)  # Cap delta time
        self.last_time = current_time
        
        if self.is_running:
            # Calculate magnetic forces
            magnetic_force = self.calculate_magnetic_force(self.train_pos)
            
            # Update physics
            self.train_acceleration = magnetic_force - (self.track_resistance * self.train_speed)
            self.train_speed += self.train_acceleration * delta_time
            self.train_speed = np.clip(self.train_speed, -MAX_SPEED, MAX_SPEED)
            self.train_pos += self.train_speed * delta_time
            
            # Track looping
            if abs(self.train_pos) > TRACK_LENGTH/2:
                self.train_pos = -TRACK_LENGTH/2 if self.train_pos > 0 else TRACK_LENGTH/2
            
            # Record history for trail
            self.train_history.append((self.train_pos, current_time))
            
            # Update camera target to follow train in third person view
            if self.third_person_view:
                self.camera_target[0] = self.train_pos
        
        # FPS calculation
        self.frame_count += 1
        if self.frame_count >= 10:
            self.fps = self.frame_count / (current_time - (self.last_time - delta_time * self.frame_count))
            self.frame_count = 0
    
    def set_shader(self, name: str):
        """Activate a shader program"""
        if not self.use_shaders or name == 'default' or name not in self.shaders:
            glUseProgram(0)
            self.current_shader = None
            return
        
        shader_program = self.shaders[name]
        if shader_program is not None:
            glUseProgram(shader_program)
            self.current_shader = name
            
            # Set common uniforms
            time_loc = glGetUniformLocation(shader_program, "time")
            if time_loc != -1:
                glUniform1f(time_loc, time.time())
    
    def draw_train(self):
        """Render the maglev train with advanced effects"""
        self.set_shader('train')
        
        # Calculate hover height - only apply when train is moving
        hover_height = 0.5
        if abs(self.train_speed) > 0.1:
            hover_height += 0.05 * math.sin(time.time() * 3)
        
        glPushMatrix()
        glTranslatef(self.train_pos, hover_height, 0)
        
        # Use the 3D model if available, otherwise fall back to procedural geometry
        if self.train_model:
            glCallList(self.train_model)
        else:
            # Fallback to procedural geometry
            # Train body - more streamlined shape
            glColor3f(0.7, 0.1, 0.1)  # Deeper red for main body
            
            # Draw solid train body
            self.draw_box((-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2), 
                         (TRAIN_LENGTH, TRAIN_HEIGHT, TRAIN_WIDTH))
            
            # Windows - more realistic with metallic frames
            glColor4f(0.2, 0.7, 0.9, 0.8)  # Blue tinted glass
            for i in range(4):
                offset = -TRAIN_LENGTH/2 + 0.4 + (i * (TRAIN_LENGTH-0.8)/4)
                
                # Window on left side
                glPushMatrix()
                glTranslatef(offset + 0.2, 0.1, -TRAIN_WIDTH/2)
                glScalef(0.4, 0.3, 0.05)
                self.draw_cube(1.0)
                glPopMatrix()
                
                # Window on right side
                glPushMatrix()
                glTranslatef(offset + 0.2, 0.1, TRAIN_WIDTH/2)
                glScalef(0.4, 0.3, 0.05)
                self.draw_cube(1.0)
                glPopMatrix()
            
            # Front headlight
            glColor3f(1.0, 1.0, 0.8)  # Light yellow
            glPushMatrix()
            glTranslatef(-TRAIN_LENGTH/2 + 0.01, 0, 0)
            glScalef(0.02, 0.2, 0.2)
            self.draw_cube(1.0)
            glPopMatrix()
            
            # Back taillight
            glColor3f(1.0, 0.2, 0.1)  # Red
            glPushMatrix()
            glTranslatef(TRAIN_LENGTH/2 - 0.01, 0, 0)
            glScalef(0.02, 0.2, 0.2)
            self.draw_cube(1.0)
            glPopMatrix()
        
        # Levitation effect - only visible when moving or starting
        if self.is_running or abs(self.train_speed) > 0.05:
            self.set_shader('magnetic_field')
            glColor4f(0.2, 0.8, 0.9, 0.7)
            
            # Use display list for optimization if many instances
            for i in range(6):
                offset = -TRAIN_LENGTH/2 + 0.3 + (i * (TRAIN_LENGTH-0.6)/5)
                glPushMatrix()
                glTranslatef(offset, -TRAIN_HEIGHT/2 - 0.15, 0)
                self.draw_sphere(0.12, 12, 12)  # Reduced polygon count
                glPopMatrix()
        
        glPopMatrix()
        self.set_shader('default')
        
    def draw_rounded_box(self, pos, size, radius):
        """Draw a box with slightly rounded edges"""
        x, y, z = pos
        w, h, d = size
        
        # Main body
        glBegin(GL_QUADS)
        # Front
        glNormal3f(-1, 0, 0)
        glVertex3f(x, y+radius, z+radius)
        glVertex3f(x, y+h-radius, z+radius)
        glVertex3f(x, y+h-radius, z+d-radius)
        glVertex3f(x, y+radius, z+d-radius)
        
        # Back
        glNormal3f(1, 0, 0)
        glVertex3f(x+w, y+radius, z+radius)
        glVertex3f(x+w, y+h-radius, z+radius)
        glVertex3f(x+w, y+h-radius, z+d-radius)
        glVertex3f(x+w, y+radius, z+d-radius)
        
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(x+radius, y+h, z+radius)
        glVertex3f(x+w-radius, y+h, z+radius)
        glVertex3f(x+w-radius, y+h, z+d-radius)
        glVertex3f(x+radius, y+h, z+d-radius)
        
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(x+radius, y, z+radius)
        glVertex3f(x+w-radius, y, z+radius)
        glVertex3f(x+w-radius, y, z+d-radius)
        glVertex3f(x+radius, y, z+d-radius)
        
        # Left
        glNormal3f(0, 0, -1)
        glVertex3f(x+radius, y, z)
        glVertex3f(x+w-radius, y, z)
        glVertex3f(x+w-radius, y+h, z)
        glVertex3f(x+radius, y+h, z)
        
        # Right
        glNormal3f(0, 0, 1)
        glVertex3f(x+radius, y, z+d)
        glVertex3f(x+w-radius, y, z+d)
        glVertex3f(x+w-radius, y+h, z+d)
        glVertex3f(x+radius, y+h, z+d)
        glEnd()
    
    def draw_track(self):
        """Render the track with magnets"""
        # Calculate visible segment of track based on camera position
        # This optimization only renders magnets that are in view
        visible_range = 30.0  # Adjust based on camera distance
        min_x = self.train_pos - visible_range
        max_x = self.train_pos + visible_range
        
        # Draw main track structure
        glColor3f(0.4, 0.4, 0.4)  # Lighter gray for main track
        
        # Draw track base (continuous structure)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-TRACK_LENGTH/2, -0.2, -1.0)
        glVertex3f(TRACK_LENGTH/2, -0.2, -1.0)
        glVertex3f(TRACK_LENGTH/2, -0.2, 1.0)
        glVertex3f(-TRACK_LENGTH/2, -0.2, 1.0)
        glEnd()
        
        # Draw track sides
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        # Left side
        glNormal3f(0, 0, 1)
        glVertex3f(-TRACK_LENGTH/2, -0.2, -1.0)
        glVertex3f(TRACK_LENGTH/2, -0.2, -1.0)
        glVertex3f(TRACK_LENGTH/2, 0.0, -1.0)
        glVertex3f(-TRACK_LENGTH/2, 0.0, -1.0)
        
        # Right side
        glNormal3f(0, 0, -1)
        glVertex3f(-TRACK_LENGTH/2, -0.2, 1.0)
        glVertex3f(TRACK_LENGTH/2, -0.2, 1.0)
        glVertex3f(TRACK_LENGTH/2, 0.0, 1.0)
        glVertex3f(-TRACK_LENGTH/2, 0.0, 1.0)
        glEnd()
        
        # Draw track rails
        glColor3f(0.6, 0.6, 0.6)  # Metallic color for rails
        glLineWidth(3.0)
        
        # Left rail
        glBegin(GL_LINE_STRIP)
        for vertex in self.track_vertices:
            x, y, _ = vertex
            glVertex3f(x, 0.0, -0.8)
        glEnd()
        
        # Right rail
        glBegin(GL_LINE_STRIP)
        for vertex in self.track_vertices:
            x, y, _ = vertex
            glVertex3f(x, 0.0, 0.8)
        glEnd()
        
        # Draw magnets - only those in visible range
        num_magnets = int(TRACK_LENGTH / MAGNET_SPACING) + 1
        
        # Create a display list for magnets if not already created
        if not hasattr(self, 'magnet_display_list'):
            self.magnet_display_list = glGenLists(1)
            glNewList(self.magnet_display_list, GL_COMPILE)
            
            # Magnet base
            glColor3f(0.1, 0.1, 0.5)
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            glVertex3f(-0.4, -0.05, -0.4)
            glVertex3f(0.4, -0.05, -0.4)
            glVertex3f(0.4, -0.05, 0.4)
            glVertex3f(-0.4, -0.05, 0.4)
            glEnd()
            
            # North pole (red)
            glColor3f(0.9, 0.1, 0.1)
            glPushMatrix()
            glTranslatef(0, 0.05, 0.3)
            self.draw_cube(0.4)
            glPopMatrix()
            
            # South pole (blue)
            glColor3f(0.1, 0.1, 0.9)
            glPushMatrix()
            glTranslatef(0, 0.05, -0.3)
            self.draw_cube(0.4)
            glPopMatrix()
            
            glEndList()
        
        # Draw only visible magnets
        self.set_shader('magnetic_field')
        for i in range(num_magnets):
            magnet_x = -TRACK_LENGTH/2 + i * MAGNET_SPACING
            
            # Skip if outside visible range
            if magnet_x < min_x or magnet_x > max_x:
                continue
                
            glPushMatrix()
            glTranslatef(magnet_x, 0, 0)
            glCallList(self.magnet_display_list)
            glPopMatrix()
        
        self.set_shader('default')
    
    def draw_field_visualization(self):
        """Render magnetic field visualization"""
        if not (self.show_field_lines or self.show_field_arrows):
            return
        
        # Optimization: Only draw field effects near the train
        visible_range = 25.0
        min_x = self.train_pos - visible_range
        max_x = self.train_pos + visible_range
        
        # Adjust alpha based on train speed instead of oscillation when stationary
        base_alpha = 0.3
        if self.is_running:
            speed_factor = min(1.0, abs(self.train_speed) / 10.0)
            alpha = base_alpha + 0.4 * speed_factor
        else:
            alpha = base_alpha
        
        self.set_shader('magnetic_field')
        
        # Field lines - only draw those in visible range
        if self.show_field_lines:
            glColor4f(0.8, 0.8, 0.2, alpha)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            for i in range(0, len(self.field_lines), 2):
                x1 = self.field_lines[i][0]
                
                # Skip if outside visible range
                if x1 < min_x or x1 > max_x:
                    continue
                
                # Apply field distortion based on train movement
                x1, y1, z1 = self.field_lines[i]
                x2, y2, z2 = self.field_lines[i+1]
                
                # Only distort field lines when train is moving
                if self.is_running and abs(self.train_speed) > 1.0:
                    # Calculate distortion based on train speed and direction
                    field_intensity = min(1.0, abs(self.train_speed) / 10.0) * 0.3
                    distortion = math.sin(time.time() * 3 + x1 * 0.5) * field_intensity
                    direction = 1 if self.train_speed > 0 else -1
                    
                    # Apply distortion to field lines
                    y1 += distortion * 0.1
                    y2 += distortion * 0.2
                    x1 += direction * distortion * 0.05
                    x2 += direction * distortion * 0.1
                
                glVertex3f(x1, y1, z1)
                glVertex3f(x2, y2, z2)
            glEnd()
        
        # Field arrows - only draw those in visible range
        if self.show_field_arrows:
            glColor4f(0.2, 0.8, 0.8, alpha)
            
            # Use fewer arrows when train is moving fast to reduce GPU load
            arrow_skip = 1
            if abs(self.train_speed) > 10:
                arrow_skip = 2
            if abs(self.train_speed) > 20:
                arrow_skip = 3
                
            for i, arrow in enumerate(self.field_arrows):
                # Skip some arrows based on speed
                if i % arrow_skip != 0:
                    continue
                    
                x, y, z, dx, dy, dz = arrow
                
                # Skip if outside visible range
                if x < min_x or x > max_x:
                    continue
                
                # Apply field distortion based on train movement
                if self.is_running and abs(self.train_speed) > 1.0:
                    # Calculate distortion based on train speed and direction
                    field_intensity = min(1.0, abs(self.train_speed) / 10.0) * 0.5
                    distortion = math.sin(time.time() * 2 + x * 0.5) * field_intensity
                    direction = 1 if self.train_speed > 0 else -1
                    
                    # Modify arrow direction based on train movement
                    dy += distortion * 0.2
                    dx += direction * distortion * 0.1
                
                glPushMatrix()
                glTranslatef(x, y, z)
                
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(dx, dy, dz)
                glEnd()
                
                # Only draw cone for closer arrows to improve performance
                if abs(x - self.train_pos) < 10:
                    glTranslatef(dx, dy, dz)
                    glRotatef(math.degrees(math.atan2(dz, dx)), 0, 1, 0)
                    self.draw_cone(0.1, 0.3)
                
                glPopMatrix()
        
        # Dynamic field lines near train when moving
        if self.is_running and abs(self.train_speed) > 1.0:
            # Scale effect with speed
            intensity = min(1.0, abs(self.train_speed) / 15.0)
            glColor4f(0.9, 0.2, 0.9, 0.7 * intensity)
            
            # Reduce number of lines at higher speeds
            step = 0.5
            if abs(self.train_speed) > 15:
                step = 0.75
                
            glBegin(GL_LINES)
            for i in np.arange(-2, 2, step):
                x = self.train_pos + i
                
                # Use train speed for animation instead of time
                phase = (self.train_pos * 0.5) + i * 2
                amplitude = 0.4 * math.sin(phase) * intensity
                
                glVertex3f(x, 0.1, -1.5)
                glVertex3f(x + amplitude, 0.1 + abs(amplitude), 0)
                
                glVertex3f(x, 0.1, 1.5)
                glVertex3f(x + amplitude, 0.1 + abs(amplitude), 0)
            glEnd()
        
        self.set_shader('default')
    
    def draw_train_trail(self):
        """Render the train's movement trail"""
        if not self.show_trail or not self.train_history:
            return
        
        current_time = time.time()
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for pos, t in self.train_history:
            age = current_time - t
            if age < 5.0:  # Only show recent history
                alpha = 1.0 - age / 5.0
                glColor4f(0.8, 0.8, 0.2, alpha)
                glVertex3f(pos, 0.5, 0)
        glEnd()
    
    def draw_sphere(self, radius: float, slices: int = 16, stacks: int = 16):
        """Draw a sphere with normals for lighting"""
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + (i) / stacks)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + (i+1) / stacks)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * (j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                
                glNormal3f(x * zr0, y * zr0, z0)
                glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
                glNormal3f(x * zr1, y * zr1, z1)
                glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
            glEnd()
    
    def draw_cube(self, size: float):
        """Draw a cube with normals for lighting"""
        hs = size / 2
        glBegin(GL_QUADS)
        
        # Front
        glNormal3f(0, 0, 1)
        glVertex3f(-hs, -hs, hs)
        glVertex3f(hs, -hs, hs)
        glVertex3f(hs, hs, hs)
        glVertex3f(-hs, hs, hs)
        
        # Back
        glNormal3f(0, 0, -1)
        glVertex3f(-hs, -hs, -hs)
        glVertex3f(-hs, hs, -hs)
        glVertex3f(hs, hs, -hs)
        glVertex3f(hs, -hs, -hs)
        
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(-hs, hs, -hs)
        glVertex3f(-hs, hs, hs)
        glVertex3f(hs, hs, hs)
        glVertex3f(hs, hs, -hs)
        
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(-hs, -hs, -hs)
        glVertex3f(hs, -hs, -hs)
        glVertex3f(hs, -hs, hs)
        glVertex3f(-hs, -hs, hs)
        
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(hs, -hs, -hs)
        glVertex3f(hs, hs, -hs)
        glVertex3f(hs, hs, hs)
        glVertex3f(hs, -hs, hs)
        
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(-hs, -hs, -hs)
        glVertex3f(-hs, -hs, hs)
        glVertex3f(-hs, hs, hs)
        glVertex3f(-hs, hs, -hs)
        
        glEnd()
    
    def draw_cone(self, base: float, height: float, slices: int = 16):
        """Draw a cone with normals for lighting"""
        # Base
        glBegin(GL_POLYGON)
        glNormal3f(0, 0, -1)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = math.cos(angle) * base
            y = math.sin(angle) * base
            glVertex3f(x, y, 0)
        glEnd()
        
        # Side
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, height)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = math.cos(angle) * base
            y = math.sin(angle) * base
            normal = [math.cos(angle), math.sin(angle), base/height]
            norm_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            glNormal3f(normal[0]/norm_len, normal[1]/norm_len, normal[2]/norm_len)
            glVertex3f(x, y, 0)
        glEnd()
    
    def render_text(self, text: str, position: Tuple[int, int], color=(255, 255, 255)):
        """Render text to screen using pygame's font system"""
        font = pygame.font.SysFont('Arial', 20, bold=True)
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        glWindowPos2d(position[0], position[1])
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), 
                    GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def draw_hud(self):
        """Render the heads-up display with controls and info"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Simulation info
        if self.show_debug:
            self.render_text(f"Speed: {abs(self.train_speed):.1f} m/s", (20, WINDOW_HEIGHT - 30))
            self.render_text(f"Position: {self.train_pos:.1f} m", (20, WINDOW_HEIGHT - 60))
            self.render_text(f"Acceleration: {self.train_acceleration:.2f} m/s²", (20, WINDOW_HEIGHT - 90))
            self.render_text(f"Field Strength: {self.field_strength:.1f}", (20, WINDOW_HEIGHT - 120))
            self.render_text(f"FPS: {self.fps:.1f}", (20, WINDOW_HEIGHT - 150))
        
        # Controls help
        if self.show_help:
            controls = [
                "Controls:",
                "+ / - Keys - Increase/Decrease Speed",
                "Space - Start/Stop Simulation",
                "R - Reset Simulation",
                "Left Click+Drag - Rotate View",
                "Mouse Wheel - Zoom In/Out",
                "F - Toggle Field Lines",
                "T - Toggle Train Trail",
                "H - Toggle Help",
                "D - Toggle Debug Info"
            ]
            
            for i, line in enumerate(controls):
                y_pos = 20 + i * 25
                self.render_text(line, (WINDOW_WIDTH - 400, y_pos))
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render(self):
        """Main rendering function"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 200.0)
        
        # Setup camera for third-person view
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Calculate camera position based on angles and distance
        cam_x = self.camera_target[0] + self.camera_distance * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        cam_y = self.camera_target[1] + self.camera_distance * math.sin(math.radians(self.camera_angle_x))
        cam_z = self.camera_target[2] + self.camera_distance * math.cos(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        
        # Look at the target point
        gluLookAt(
            cam_x, cam_y, cam_z,
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0, 1, 0
        )
        
        # Coordinate axes
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(5, 0, 0)  # X
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 5, 0)  # Y
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 5)  # Z
        glEnd()
        
        # Main rendering
        self.draw_track()
        self.draw_field_visualization()
        self.draw_train_trail()
        self.draw_train()
        
        # HUD
        self.draw_hud()
        
        # Reset shader
        self.set_shader('default')
    
    # Simple toggle functions without menu dependencies
    def toggle_field_lines(self):
        """Toggle field lines visibility"""
        self.show_field_lines = not self.show_field_lines
        
    def toggle_field_arrows(self):
        """Toggle field arrows visibility"""
        self.show_field_arrows = not self.show_field_arrows
        
    def toggle_train_trail(self):
        """Toggle train trail visibility"""
        self.show_trail = not self.show_trail
        
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.train_pos = -TRACK_LENGTH/2
        self.train_speed = 0.0
        self.is_running = False
        self.train_history.clear()
        
    def handle_events(self) -> bool:
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_running = not self.is_running
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    # Use + key to increase speed
                    self.train_speed += ACCELERATION
                    self.train_speed = min(MAX_SPEED, self.train_speed)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    # Use - key to decrease speed
                    self.train_speed -= ACCELERATION
                    self.train_speed = max(-MAX_SPEED, self.train_speed)
                elif event.key == pygame.K_r:
                    self.train_pos = -TRACK_LENGTH/2
                    self.train_speed = 0.0
                    self.is_running = False
                    self.train_history.clear()
                elif event.key == pygame.K_f:
                    self.show_field_lines = not self.show_field_lines
                elif event.key == pygame.K_a:
                    self.show_field_arrows = not self.show_field_arrows
                elif event.key == pygame.K_t:
                    self.show_trail = not self.show_trail
                elif event.key == pygame.K_s:
                    self.use_shaders = not self.use_shaders
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_ESCAPE:
                    return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse
                    self.mouse_dragging = True
                    self.mouse_last_pos = event.pos
                elif event.button == 4:  # Scroll up
                    self.camera_distance = max(5, self.camera_distance - 2)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(100, self.camera_distance + 2)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.mouse_last_pos[0]
                    dy = event.pos[1] - self.mouse_last_pos[1]
                    
                    # Free mode - look around
                    self.camera_angle_y += dx * 0.5
                    self.camera_angle_x = np.clip(self.camera_angle_x + dy * 0.5, -89, 89)
                    
                    self.mouse_last_pos = event.pos
        
        # Update camera target to follow train in third person view
        if self.third_person_view:
            self.camera_target[0] = self.train_pos
        
        return True
    
    def create_display_lists(self):
        """Create OpenGL display lists for static geometry to improve performance"""
        # Create display list for a single magnet
        self.magnet_display_list = glGenLists(1)
        glNewList(self.magnet_display_list, GL_COMPILE)
        
        # Magnet base
        glColor3f(0.1, 0.1, 0.5)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(-0.4, -0.05, -0.4)
        glVertex3f(0.4, -0.05, -0.4)
        glVertex3f(0.4, -0.05, 0.4)
        glVertex3f(-0.4, -0.05, 0.4)
        glEnd()
        
        # North pole (red)
        glColor3f(0.9, 0.1, 0.1)
        glPushMatrix()
        glTranslatef(0, 0.05, 0.3)
        self.draw_cube(0.4)
        glPopMatrix()
        
        # South pole (blue)
        glColor3f(0.1, 0.1, 0.9)
        glPushMatrix()
        glTranslatef(0, 0.05, -0.3)
        self.draw_cube(0.4)
        glPopMatrix()
        
        glEndList()
    def load_train_model(self):
        """Load a 3D model for the train"""
        try:
            from OpenGL.GL import glCallList, glGenLists, glNewList, glEndList, GL_COMPILE
            
            # Create a display list for the train model
            self.train_model = glGenLists(1)
            glNewList(self.train_model, GL_COMPILE)
            
            # This is a simplified train model built with solid geometry
            # Front section (nose)
            glColor3f(0.7, 0.1, 0.1)
            self.draw_tapered_box(
                (-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2),
                (TRAIN_LENGTH/5, TRAIN_HEIGHT, TRAIN_WIDTH),
                front_taper=0.7
            )
            
            # Middle section (main body)
            glColor3f(0.7, 0.1, 0.1)
            self.draw_box(
                (-TRAIN_LENGTH/2 + TRAIN_LENGTH/5, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2),
                (TRAIN_LENGTH*3/5, TRAIN_HEIGHT, TRAIN_WIDTH)
            )
            
            # Back section (tapered)
            glColor3f(0.7, 0.1, 0.1)
            self.draw_tapered_box(
                (TRAIN_LENGTH/2 - TRAIN_LENGTH/5, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2),
                (TRAIN_LENGTH/5, TRAIN_HEIGHT, TRAIN_WIDTH),
                back_taper=0.7
            )
            
            # Top section (aerodynamic)
            glColor3f(0.8, 0.2, 0.2)
            self.draw_tapered_box(
                (-TRAIN_LENGTH/2 + TRAIN_LENGTH/10, TRAIN_HEIGHT/2 - 0.1, -TRAIN_WIDTH/2 + 0.1),
                (TRAIN_LENGTH*4/5, TRAIN_HEIGHT/4, TRAIN_WIDTH - 0.2),
                front_taper=0.5, back_taper=0.5
            )
            
            # Windows
            glColor4f(0.2, 0.7, 0.9, 0.8)
            window_height = TRAIN_HEIGHT * 0.4
            window_width = TRAIN_WIDTH * 0.05
            window_spacing = TRAIN_LENGTH / 6
            
            for i in range(5):
                offset = -TRAIN_LENGTH/2 + TRAIN_LENGTH/6 + i * window_spacing
                
                # Left side windows
                glPushMatrix()
                glTranslatef(offset, 0, -TRAIN_WIDTH/2 + window_width/2)
                glScalef(TRAIN_LENGTH/10, window_height, window_width)
                self.draw_cube(1.0)
                glPopMatrix()
                
                # Right side windows
                glPushMatrix()
                glTranslatef(offset, 0, TRAIN_WIDTH/2 - window_width/2)
                glScalef(TRAIN_LENGTH/10, window_height, window_width)
                self.draw_cube(1.0)
                glPopMatrix()
            
            # Bottom details
            glColor3f(0.3, 0.3, 0.3)
            for i in range(3):
                offset = -TRAIN_LENGTH/3 + i * (TRAIN_LENGTH/3)
                
                # Undercarriage
                glPushMatrix()
                glTranslatef(offset, -TRAIN_HEIGHT/2 - 0.1, 0)
                glScalef(TRAIN_LENGTH/4, 0.1, TRAIN_WIDTH*0.8)
                self.draw_cube(1.0)
                glPopMatrix()
            
            # Front and back details
            # Headlights
            glColor3f(1.0, 1.0, 0.8)
            glPushMatrix()
            glTranslatef(-TRAIN_LENGTH/2 + 0.05, 0, 0)
            glScalef(0.05, 0.3, 0.6)
            self.draw_cube(1.0)
            glPopMatrix()
            
            # Taillights
            glColor3f(1.0, 0.2, 0.1)
            glPushMatrix()
            glTranslatef(TRAIN_LENGTH/2 - 0.05, 0, 0)
            glScalef(0.05, 0.3, 0.6)
            self.draw_cube(1.0)
            glPopMatrix()
            
            glEndList()
            return True
            
        except Exception as e:
            print(f"Error creating train model: {e}")
            return False
            
    def draw_tapered_box(self, pos, size, front_taper=0.0, back_taper=0.0):
        """Draw a box with tapered ends for more streamlined appearance"""
        x, y, z = pos
        w, h, d = size
        
        # Calculate taper dimensions
        front_h = h * (1.0 - front_taper)
        front_d = d * (1.0 - front_taper)
        back_h = h * (1.0 - back_taper)
        back_d = d * (1.0 - back_taper)
        
        glBegin(GL_QUADS)
        
        # Front face (tapered)
        glNormal3f(-1, 0, 0)
        glVertex3f(x, y, z)
        glVertex3f(x, y + front_h, z + (d - front_d)/2)
        glVertex3f(x, y + front_h, z + d - (d - front_d)/2)
        glVertex3f(x, y, z + d)
        
        # Back face (tapered)
        glNormal3f(1, 0, 0)
        glVertex3f(x + w, y, z)
        glVertex3f(x + w, y + back_h, z + (d - back_d)/2)
        glVertex3f(x + w, y + back_h, z + d - (d - back_d)/2)
        glVertex3f(x + w, y, z + d)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(x, y + front_h, z + (d - front_d)/2)
        glVertex3f(x + w, y + back_h, z + (d - back_d)/2)
        glVertex3f(x + w, y + back_h, z + d - (d - back_d)/2)
        glVertex3f(x, y + front_h, z + d - (d - front_d)/2)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(x, y, z)
        glVertex3f(x + w, y, z)
        glVertex3f(x + w, y, z + d)
        glVertex3f(x, y, z + d)
        
        # Left face
        glNormal3f(0, 0, -1)
        glVertex3f(x, y, z)
        glVertex3f(x, y + front_h, z + (d - front_d)/2)
        glVertex3f(x + w, y + back_h, z + (d - back_d)/2)
        glVertex3f(x + w, y, z)
        
        # Right face
        glNormal3f(0, 0, 1)
        glVertex3f(x, y, z + d)
        glVertex3f(x, y + front_h, z + d - (d - front_d)/2)
        glVertex3f(x + w, y + back_h, z + d - (d - back_d)/2)
        glVertex3f(x + w, y, z + d)
        
        glEnd()
        
    def draw_box(self, pos, size):
        """Draw a solid box with no gaps"""
        x, y, z = pos
        w, h, d = size
        
        glBegin(GL_QUADS)
        # Front
        glNormal3f(-1, 0, 0)
        glVertex3f(x, y, z)
        glVertex3f(x, y + h, z)
        glVertex3f(x, y + h, z + d)
        glVertex3f(x, y, z + d)
        
        # Back
        glNormal3f(1, 0, 0)
        glVertex3f(x + w, y, z)
        glVertex3f(x + w, y + h, z)
        glVertex3f(x + w, y + h, z + d)
        glVertex3f(x + w, y, z + d)
        
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(x, y + h, z)
        glVertex3f(x + w, y + h, z)
        glVertex3f(x + w, y + h, z + d)
        glVertex3f(x, y + h, z + d)
        
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(x, y, z)
        glVertex3f(x + w, y, z)
        glVertex3f(x + w, y, z + d)
        glVertex3f(x, y, z + d)
        
        # Left
        glNormal3f(0, 0, -1)
        glVertex3f(x, y, z)
        glVertex3f(x + w, y, z)
        glVertex3f(x + w, y + h, z)
        glVertex3f(x, y + h, z)
        
        # Right
        glNormal3f(0, 0, 1)
        glVertex3f(x, y, z + d)
        glVertex3f(x + w, y, z + d)
        glVertex3f(x + w, y + h, z + d)
        glVertex3f(x, y + h, z + d)
        glEnd()
    
    def process_keys(self):
        """Process keyboard input for continuous actions"""
        keys = pygame.key.get_pressed()
        
        # Camera movement with arrow keys in free mode
        move_speed = 0.5
        if keys[pygame.K_UP]:
            self.camera_pos[2] -= move_speed
        if keys[pygame.K_DOWN]:
            self.camera_pos[2] += move_speed
        if keys[pygame.K_LEFT]:
            self.camera_pos[0] -= move_speed
        if keys[pygame.K_RIGHT]:
            self.camera_pos[0] += move_speed
    
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_events()
            self.process_keys()  # Process continuous key presses
            self.update()
            self.render()
            pygame.display.flip()
            clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    simulation = MaglevSimulation()
    simulation.run()