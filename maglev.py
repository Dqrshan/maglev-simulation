import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import time
from collections import deque

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
TRAIN_LENGTH = 2.0
TRAIN_WIDTH = 1.0
TRAIN_HEIGHT = 0.5
TRACK_LENGTH = 40.0
MAGNET_SPACING = 1.5
FIELD_STRENGTH = 1.0
MAX_SPEED = 30.0
ACCELERATION = 0.5

class MaglevSimulation:
    def __init__(self):
        pygame.init()        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF|OPENGL)
        pygame.display.set_caption("Maglev Simulation with Magnetic Field Visualization")
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20, bold=True)
        self.train_pos = -TRACK_LENGTH/2
        self.train_speed = 0.0
        self.is_running = False
        self.camera_distance = 20.0
        self.camera_angle_x = 30.0
        self.camera_angle_y = -30.0
        self.camera_pos = [0, 5, 0]
        self.mouse_dragging = False
        self.mouse_last_pos = (0, 0)
        self.field_lines = []
        self.field_arrows = []
        self.generate_field_geometry()
        self.track_vertices = []
        self.generate_track_geometry()
        self.train_history = deque(maxlen=100)
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.setup_opengl()
        
    def setup_opengl(self):
        """Set up OpenGL rendering context"""
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1, 1, 1, 1])
        glMaterialfv(GL_FRONT, GL_SHININESS, [50])
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        
    def generate_track_geometry(self):
        """Generate vertices for the maglev track"""
        segments = 100
        for i in range(segments + 1):
            x = -TRACK_LENGTH/2 + (i / segments) * TRACK_LENGTH
            self.track_vertices.append((x, 0, 0))
    
    def generate_field_geometry(self):
        """Generate magnetic field lines and arrows for visualization"""
        num_magnets = int(TRACK_LENGTH / MAGNET_SPACING) + 1
        line_resolution = 20
        
        for i in range(num_magnets):
            magnet_x = -TRACK_LENGTH/2 + i * MAGNET_SPACING
            
            for j in range(line_resolution):
                angle1 = 2 * math.pi * j / line_resolution
                angle2 = 2 * math.pi * (j + 1) / line_resolution
                radius = 0.8
                
                x1 = magnet_x + radius * math.cos(angle1)
                z1 = radius * math.sin(angle1)
                x2 = magnet_x + radius * math.cos(angle2)
                z2 = radius * math.sin(angle2)
                
                self.field_lines.append((x1, 0.1, z1))
                self.field_lines.append((x2, 0.1, z2))
            
            for direction in [-1, 1]:
                for step in np.linspace(0.2, 1.5, 5):
                    x = magnet_x
                    z = direction * step
                    self.field_arrows.append((x, 0.1, z, 0, 0, direction * 0.2))
    
    def update(self):
        """Update simulation state"""
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        
        if self.is_running:
            self.train_pos += self.train_speed * delta_time
            if abs(self.train_pos) > TRACK_LENGTH/2:
                self.train_pos = -TRACK_LENGTH/2 if self.train_pos > 0 else TRACK_LENGTH/2
            
            self.train_history.append((self.train_pos, current_time))
        
        self.frame_count += 1
        if self.frame_count >= 10:
            self.fps = self.frame_count / (current_time - (self.last_time - delta_time * self.frame_count))
            self.frame_count = 0
    
    def draw_cube(self, size):
        """Draw a cube without using glutSolidCube"""
        size = size / 2
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, -size, -size)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(-size, -size, size)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(size, -size, size)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, size, -size)
        
        glEnd()
    
    def draw_sphere(self, radius, slices=10, stacks=10):
        """Draw a sphere without using glutSolidSphere"""
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
    
    def draw_cone(self, base, height, slices=10, stacks=10):
        """Draw a cone without using glutSolidCone"""
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, height)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = math.cos(angle) * base
            y = math.sin(angle) * base
            glNormal3f(math.cos(angle), math.sin(angle), base/height)
            glVertex3f(x, y, 0)
        glEnd()
        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = math.cos(angle) * base
            y = math.sin(angle) * base
            glNormal3f(0, 0, -1)
            glVertex3f(x, y, 0)
        glEnd()
    
    def draw_train(self):
        """Draw the maglev train"""
        glPushMatrix()
        glTranslatef(self.train_pos, 0.5 + 0.1 * math.sin(time.time() * 5), 0)
        
        # Train body
        glColor3f(0.8, 0.2, 0.2)
        glBegin(GL_QUADS)
        # Front
        glNormal3f(-1, 0, 0)
        glVertex3f(-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        # Back
        glNormal3f(1, 0, 0)
        glVertex3f(TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(-TRAIN_LENGTH/2, TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        # Left
        glNormal3f(0, 0, -1)
        glVertex3f(-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, TRAIN_HEIGHT/2, -TRAIN_WIDTH/2)
        # Right
        glNormal3f(0, 0, 1)
        glVertex3f(-TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, -TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(TRAIN_LENGTH/2, TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glVertex3f(-TRAIN_LENGTH/2, TRAIN_HEIGHT/2, TRAIN_WIDTH/2)
        glEnd()
        
        # Windows
        glColor3f(0.7, 0.9, 1.0)
        for i in range(3):
            offset = -TRAIN_LENGTH/3 + (i * TRAIN_LENGTH/3)
            glBegin(GL_QUADS)
            glNormal3f(0, 0, -1)
            glVertex3f(offset - 0.3, 0.1, -TRAIN_WIDTH/2 + 0.01)
            glVertex3f(offset + 0.3, 0.1, -TRAIN_WIDTH/2 + 0.01)
            glVertex3f(offset + 0.3, 0.4, -TRAIN_WIDTH/2 + 0.01)
            glVertex3f(offset - 0.3, 0.4, -TRAIN_WIDTH/2 + 0.01)
            glEnd()
        
        # Levitation effect
        glColor3f(0.2, 0.8, 0.9)
        for i in range(4):
            offset = -TRAIN_LENGTH/2.5 + (i * TRAIN_LENGTH/3.5)
            glPushMatrix()
            glTranslatef(offset, -TRAIN_HEIGHT/2 - 0.2, 0)
            self.draw_sphere(0.1)
            glPopMatrix()
        
        glPopMatrix()
    
    def draw_track(self):
        """Draw the maglev track"""
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_LINE_STRIP)
        for vertex in self.track_vertices:
            glVertex3fv(vertex)
        glEnd()
        
        num_magnets = int(TRACK_LENGTH / MAGNET_SPACING) + 1
        for i in range(num_magnets):
            magnet_x = -TRACK_LENGTH/2 + i * MAGNET_SPACING
            glPushMatrix()
            glTranslatef(magnet_x, 0, 0)
            
            # Magnet base
            glColor3f(0.1, 0.1, 0.5)
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            glVertex3f(-0.3, -0.1, -0.3)
            glVertex3f(0.3, -0.1, -0.3)
            glVertex3f(0.3, -0.1, 0.3)
            glVertex3f(-0.3, -0.1, 0.3)
            glEnd()
            
            glColor3f(0.9, 0.1, 0.1)  # North (red)
            glPushMatrix()
            glTranslatef(0, 0.1, 0.2)
            self.draw_cube(0.4)
            glPopMatrix()
            
            glColor3f(0.1, 0.1, 0.9)  # South (blue)
            glPushMatrix()
            glTranslatef(0, 0.1, -0.2)
            self.draw_cube(0.4)
            glPopMatrix()
            
            glPopMatrix()
    
    def draw_field_lines(self):
        """Draw the magnetic field visualization"""
        if not self.is_running:
            alpha = 0.3
        else:
            alpha = 0.5 + 0.2 * math.sin(time.time() * 5)
        
        glColor4f(0.8, 0.8, 0.2, alpha)
        glBegin(GL_LINES)
        for i in range(0, len(self.field_lines), 2):
            if i+1 < len(self.field_lines):  # Check bounds
                glVertex3fv(self.field_lines[i])
                glVertex3fv(self.field_lines[i+1])
        glEnd()
        
        glColor4f(0.2, 0.8, 0.8, alpha)
        for arrow in self.field_arrows:
            x, y, z, dx, dy, dz = arrow
            glPushMatrix()
            glTranslatef(x, y, z)
            
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(dx, dy, dz)
            glEnd()
            
            glPushMatrix()
            glTranslatef(dx, dy, dz)
            glScalef(0.1, 0.1, 0.1)
            self.draw_cone(0.5, 1.0)
            glPopMatrix()
            
            glPopMatrix()
        
        if self.is_running:
            glColor4f(0.9, 0.2, 0.9, 0.7)
            glBegin(GL_LINES)
            for i in np.arange(-2, 2, 0.5):
                x = self.train_pos + i
                phase = time.time() * 10 + i * 2
                amplitude = 0.3 * math.sin(phase)
                
                glVertex3f(x, 0.1, -1.5)
                glVertex3f(x + amplitude, 0.1 + abs(amplitude), 0)
                
                glVertex3f(x, 0.1, 1.5)
                glVertex3f(x + amplitude, 0.1 + abs(amplitude), 0)
            glEnd()
    
    def draw_train_trail(self):
        """Draw a fading trail behind the train"""
        current_time = time.time()
        glBegin(GL_LINE_STRIP)
        for pos, t in self.train_history:
            age = current_time - t
            if age < 3.0: 
                alpha = 1.0 - age / 3.0
                glColor4f(0.8, 0.8, 0.2, alpha)
                glVertex3f(pos, 0.5, 0)
        glEnd()
    
    def render_text(self, text, position, color=(255, 255, 255)):
        """Render text using the glWindowPos and glDrawPixels approach"""
        text_surface = self.font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_width, text_height = text_surface.get_size()
        
        glWindowPos2d(position[0], position[1])
        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    
    def draw_hud(self):
        """Draw heads-up display with information"""
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.render_text(f"Speed: {abs(self.train_speed):.1f} m/s", (20, WINDOW_HEIGHT - 30))
        self.render_text(f"Position: {self.train_pos:.1f} m", (20, WINDOW_HEIGHT - 60))
        self.render_text(f"Status: {'RUNNING' if self.is_running else 'STOPPED'}", (20, WINDOW_HEIGHT - 90))
        self.render_text(f"FPS: {self.fps:.1f}", (20, WINDOW_HEIGHT - 120))
        
        controls_text = "Controls: Up/Down - Speed | Space - Start/Stop | R - Reset | Left Click+Drag - Rotate | Scroll - Zoom"
        self.render_text(controls_text, (WINDOW_WIDTH//2 - 350, 20))
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render(self):
        """Main rendering function"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, WINDOW_WIDTH / WINDOW_HEIGHT, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2] + self.camera_distance,
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            0, 1, 0
        )
        
        glRotatef(self.camera_angle_x, 1, 0, 0)
        glRotatef(self.camera_angle_y, 0, 1, 0)
        
        glBegin(GL_LINES)
        glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(5, 0, 0)  # X
        glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 5, 0)  # Y
        glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 5)  # Z
        glEnd()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.draw_track()
        self.draw_field_lines()
        self.draw_train_trail()
        self.draw_train()
        
        self.draw_hud()
    
    def handle_events(self):
        """Handle Pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_running = not self.is_running
                elif event.key == pygame.K_UP:
                    self.train_speed += ACCELERATION
                    if self.train_speed > MAX_SPEED:
                        self.train_speed = MAX_SPEED
                elif event.key == pygame.K_DOWN:
                    self.train_speed -= ACCELERATION
                    if self.train_speed < -MAX_SPEED:
                        self.train_speed = -MAX_SPEED
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.train_pos = -TRACK_LENGTH/2
                    self.train_speed = 0.0
                    self.is_running = False
                    self.train_history.clear()
                elif event.key == pygame.K_ESCAPE:
                    return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = True
                    self.mouse_last_pos = event.pos
                elif event.button == 4:  # Scroll up
                    self.camera_distance = max(5, self.camera_distance - 1)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(50, self.camera_distance + 1)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.mouse_last_pos[0]
                    dy = event.pos[1] - self.mouse_last_pos[1]
                    self.camera_angle_y += dx * 0.5
                    self.camera_angle_x += dy * 0.5
                    self.mouse_last_pos = event.pos
        
        return True
    
    def run(self):
        """Main simulation loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            pygame.display.flip()
            clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    simulation = MaglevSimulation()
    simulation.run()