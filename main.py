# =============================================================================
# 🏰 ADVANCED TOWER DEFENSE GAME - Multi-Agent AI System
# =============================================================================
# A complete tower defense game where:
# - Players can place towers by clicking on the grid
# - Enemies use BFS pathfinding to navigate around towers
# - Towers shoot bullets at enemies automatically
# - Players earn money to buy more towers
# - Waves of enemies get progressively harder
# - FLYING ENEMIES bypass ground paths and require anti-air towers!
# =============================================================================

# --- STEP 1: IMPORT LIBRARIES ---
import pygame
import sys
import math
import random
from collections import deque

# --- STEP 2: INITIALIZE PYGAME ---
pygame.init()

# --- STEP 3: GAME SETTINGS ---
# Window size
GRID_COLS = 15
GRID_ROWS = 10
CELL_SIZE = 50
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 150  # Extra space for UI

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
PURPLE = (200, 0, 200)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
DARK_RED = (139, 0, 0)
LIGHT_BLUE = (135, 206, 250)
DARK_CYAN = (0, 139, 139)

# Game settings
FPS = 60
ENEMY_SPEED = 1.5  # Pixels per frame
BULLET_SPEED = 5

# Tower types with different stats
TOWER_TYPES = {
    'basic': {
        'cost': 50,
        'damage': 15,
        'range': 150,
        'cooldown': 30,
        'color': GREEN,
        'name': 'Basic Tower',
        'can_hit_air': False
    },
    'sniper': {
        'cost': 100,
        'damage': 50,
        'range': 250,
        'cooldown': 90,
        'color': PURPLE,
        'name': 'Sniper Tower',
        'can_hit_air': False
    },
    'rapid': {
        'cost': 75,
        'damage': 5,
        'range': 100,
        'cooldown': 10,
        'color': ORANGE,
        'name': 'Rapid Tower',
        'can_hit_air': False
    },
    'antiair': {
        'cost': 80,
        'damage': 25,
        'range': 200,
        'cooldown': 25,
        'color': CYAN,
        'name': 'Anti-Air Tower',
        'can_hit_air': True
    }
}

STARTING_MONEY = 150

# Enemy wave settings
ENEMY_SPAWN_INTERVAL = 90  # Frames between spawns
WAVE_SIZE = 5  # Enemies per wave
WAVE_DELAY = 180  # Frames between waves

# --- STEP 4: CREATE GAME WINDOW ---
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🏰 Advanced Tower Defense - Multi-Agent AI")
clock = pygame.time.Clock()

# Fonts
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)
font_tiny = pygame.font.Font(None, 18)

# --- STEP 5: CREATE GRID SYSTEM ---
# 0 = empty path, 1 = tower, 2 = base
grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

# Base position (bottom-right)
base_row = GRID_ROWS - 1
base_col = GRID_COLS - 1
grid[base_row][base_col] = 2

# Enemy spawn position (top-left)
spawn_row = 0
spawn_col = 0

# --- STEP 6: GAME VARIABLES ---
base_health = 100
max_base_health = 100
money = STARTING_MONEY
score = 0
game_over = False
game_won = False
wave_number = 1
enemies_in_wave = 0
total_enemies_spawned = 0
paused = False
selected_tower_type = 'basic'
selected_tower = None

# Lists to hold game objects
enemy_list = []
tower_list = []
bullet_list = []
particle_list = []

# Timers
spawn_timer = 0
wave_timer = 0

# Statistics
stats = {
    'total_kills': 0,
    'towers_placed': 0,
    'money_earned': 0,
    'highest_wave': 1,
    'flying_kills': 0  # NEW: Track flying enemy kills
}

# --- STEP 7: BFS PATHFINDING ---
def calculate_path(start_row, start_col, goal_row, goal_col, current_grid):
    """
    Use BFS to find shortest path from start to goal.
    Returns list of (row, col) positions.
    
    🧠 AI CONCEPT: BFS explores all possible moves level by level,
    guaranteeing the shortest path. This makes enemies intelligent!
    """
    rows = len(current_grid)
    cols = len(current_grid[0])
    
    # Queue stores: (row, col, path_taken)
    queue = deque()
    queue.append((start_row, start_col, [(start_row, start_col)]))
    
    # Track visited cells
    visited = set()
    visited.add((start_row, start_col))
    
    # Four directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current_row, current_col, path = queue.popleft()
        
        # Found the goal!
        if current_row == goal_row and current_col == goal_col:
            return path
        
        # Check all neighbors
        for dr, dc in directions:
            new_row = current_row + dr
            new_col = current_col + dc
            
            # Valid position check
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if (new_row, new_col) not in visited:
                    # Can only move on empty cells (0) or goal (2)
                    if current_grid[new_row][new_col] != 1:
                        visited.add((new_row, new_col))
                        new_path = path + [(new_row, new_col)]
                        queue.append((new_row, new_col, new_path))
    
    # No path exists
    return []

# Calculate initial path
enemy_path = calculate_path(spawn_row, spawn_col, base_row, base_col, grid)

# --- STEP 8: PARTICLE CLASS ---
class Particle:
    """Visual effect particles for explosions and impacts."""
    def __init__(self, x, y, color, vx=None, vy=None):
        self.x = x
        self.y = y
        self.color = color
        self.vx = vx if vx else random.uniform(-3, 3)
        self.vy = vy if vy else random.uniform(-3, 3)
        self.life = 30
        self.max_life = 30
        self.size = random.randint(2, 5)
    
    def update(self):
        """Update particle position and lifetime."""
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vy += 0.2  # Gravity
    
    def draw(self):
        """Draw particle with fading effect."""
        alpha = int(255 * (self.life / self.max_life))
        size = int(self.size * (self.life / self.max_life))
        if size > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)

# --- STEP 9: ENEMY CLASS (ENHANCED WITH FLYING ENEMIES) ---
class Enemy:
    """
    Enemy agent that follows BFS path to reach the base.
    Each enemy is independent (multi-agent system).
    """
    def __init__(self, path, health_multiplier=1, enemy_type='normal'):
        self.enemy_type = enemy_type
        self.is_flying = (enemy_type == 'flying')  # NEW: Flying flag
        
        if self.is_flying:
            # Flying enemies don't use paths - they fly directly!
            self.path = None
            self.x = spawn_col * CELL_SIZE + CELL_SIZE // 2
            self.y = spawn_row * CELL_SIZE + CELL_SIZE // 2
            self.target_x = base_col * CELL_SIZE + CELL_SIZE // 2
            self.target_y = base_row * CELL_SIZE + CELL_SIZE // 2
        else:
            self.path = path[:]
            self.path_index = 0
            self.x = path[0][1] * CELL_SIZE + CELL_SIZE // 2
            self.y = path[0][0] * CELL_SIZE + CELL_SIZE // 2
        
        # Different enemy types
        if enemy_type == 'fast':
            self.health = 30 * health_multiplier
            self.max_health = 30 * health_multiplier
            self.speed = ENEMY_SPEED * 2
            self.color = ORANGE
            self.reward = 15
            self.radius = 10
        elif enemy_type == 'tank':
            self.health = 100 * health_multiplier
            self.max_health = 100 * health_multiplier
            self.speed = ENEMY_SPEED * 0.5
            self.color = DARK_RED
            self.reward = 40
            self.radius = 15
        elif enemy_type == 'boss':
            self.health = 300 * health_multiplier
            self.max_health = 300 * health_multiplier
            self.speed = ENEMY_SPEED * 0.7
            self.color = PURPLE
            self.reward = 100
            self.radius = 20
        elif enemy_type == 'flying':  # NEW: Flying enemy stats
            self.health = 40 * health_multiplier
            self.max_health = 40 * health_multiplier
            self.speed = ENEMY_SPEED * 1.3  # Slightly faster
            self.color = LIGHT_BLUE
            self.reward = 35
            self.radius = 11
            self.hover_offset = 0  # For hovering animation
        else:  # normal
            self.health = 50 * health_multiplier
            self.max_health = 50 * health_multiplier
            self.speed = ENEMY_SPEED
            self.color = RED
            self.reward = 25
            self.radius = 12
        
        self.alive = True
    
    def move(self):
        """Move enemy along the path or directly to base if flying."""
        if self.is_flying:
            # Flying enemies move directly toward base
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > self.speed:
                self.x += (dx / distance) * self.speed
                self.y += (dy / distance) * self.speed
            else:
                self.x = self.target_x
                self.y = self.target_y
            
            # Update hover animation
            self.hover_offset = (self.hover_offset + 0.1) % (2 * math.pi)
        else:
            # Ground enemies follow path
            if self.path_index >= len(self.path) - 1:
                return
            
            target_row, target_col = self.path[self.path_index + 1]
            target_x = target_col * CELL_SIZE + CELL_SIZE // 2
            target_y = target_row * CELL_SIZE + CELL_SIZE // 2
            
            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.speed:
                self.path_index += 1
                self.x = target_x
                self.y = target_y
            else:
                self.x += (dx / distance) * self.speed
                self.y += (dy / distance) * self.speed
    
    def take_damage(self, damage):
        """Reduce health when hit."""
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            # Create death particles
            for _ in range(10):
                particle_list.append(Particle(self.x, self.y, self.color))
    
    def reached_base(self):
        """Check if enemy reached the base."""
        if self.is_flying:
            # Check distance to base
            distance = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
            return distance < 10
        else:
            return self.path_index >= len(self.path) - 1
    
    def draw(self):
        """Draw enemy on screen - IMPROVED HEALTH BAR."""
        # Calculate draw position (with hover effect for flying enemies)
        draw_y = self.y
        if self.is_flying:
            draw_y = self.y + math.sin(self.hover_offset) * 5
        
        # Draw shadow for flying enemies
        if self.is_flying:
            shadow_color = (100, 100, 100, 128)
            pygame.draw.ellipse(screen, GRAY, 
                              (int(self.x - self.radius), 
                               int(self.y + self.radius), 
                               self.radius * 2, self.radius))
        
        # Draw enemy circle
        pygame.draw.circle(screen, self.color, (int(self.x), int(draw_y)), self.radius)
        
        # Draw wings for flying enemies
        if self.is_flying:
            wing_offset = abs(math.sin(self.hover_offset * 2)) * 5
            # Left wing
            wing_left = [(int(self.x - self.radius - wing_offset), int(draw_y)),
                        (int(self.x - self.radius - 5 - wing_offset), int(draw_y - 8)),
                        (int(self.x - self.radius), int(draw_y - 3))]
            pygame.draw.polygon(screen, DARK_CYAN, wing_left)
            
            # Right wing
            wing_right = [(int(self.x + self.radius + wing_offset), int(draw_y)),
                         (int(self.x + self.radius + 5 + wing_offset), int(draw_y - 8)),
                         (int(self.x + self.radius), int(draw_y - 3))]
            pygame.draw.polygon(screen, DARK_CYAN, wing_right)
        
        # Health bar
        bar_width = max(30, self.radius * 2)
        bar_height = 6
        health_percent = max(0, min(1, self.health / self.max_health))
        
        bar_x = self.x - bar_width // 2
        bar_y = draw_y - self.radius - 15
        
        # Background (black border)
        pygame.draw.rect(screen, BLACK, (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2))
        
        # Background (dark red)
        pygame.draw.rect(screen, DARK_RED, (bar_x, bar_y, bar_width, bar_height))
        
        # Health (gradient color based on health)
        if health_percent > 0.6:
            health_color = GREEN
        elif health_percent > 0.3:
            health_color = YELLOW
        else:
            health_color = ORANGE
        
        pygame.draw.rect(screen, health_color, (bar_x, bar_y, int(bar_width * health_percent), bar_height))
        
        # Special indicators
        if self.enemy_type == 'boss':
            pygame.draw.circle(screen, YELLOW, (int(self.x), int(draw_y)), self.radius + 3, 2)
            health_text = font_tiny.render(f"{int(self.health)}", True, WHITE)
            text_rect = health_text.get_rect(center=(int(self.x), bar_y - 10))
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                outline_rect = health_text.get_rect(center=(int(self.x) + dx, bar_y - 10 + dy))
                screen.blit(health_text, outline_rect)
            screen.blit(health_text, text_rect)
        elif self.is_flying:
            # Flying indicator
            pygame.draw.circle(screen, WHITE, (int(self.x), int(draw_y)), self.radius + 2, 1)

# --- STEP 10: TOWER CLASS (ENHANCED WITH ANTI-AIR) ---
class Tower:
    """
    Tower agent that autonomously detects and shoots enemies.
    Each tower acts independently (multi-agent behavior).
    """
    def __init__(self, row, col, tower_type='basic'):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE + CELL_SIZE // 2
        self.y = row * CELL_SIZE + CELL_SIZE // 2
        self.tower_type = tower_type
        self.level = 1
        
        # Load tower stats
        self.load_stats()
        
        self.cooldown = 0
        self.target = None
    
    def load_stats(self):
        """Load stats based on tower type and level."""
        stats = TOWER_TYPES[self.tower_type]
        self.range = stats['range'] * (1 + (self.level - 1) * 0.2)
        self.damage = stats['damage'] * (1 + (self.level - 1) * 0.5)
        self.max_cooldown = stats['cooldown']
        self.color = stats['color']
        self.upgrade_cost = stats['cost'] * self.level
        self.can_hit_air = stats['can_hit_air']  # NEW: Can this tower hit flying enemies?
    
    def upgrade(self):
        """Upgrade tower to next level."""
        self.level += 1
        self.load_stats()
    
    def update(self, enemies):
        """Update tower state and shoot if possible."""
        if self.cooldown > 0:
            self.cooldown -= 1
        
        # Find closest enemy in range
        self.target = None
        min_distance = self.range + 1
        
        for enemy in enemies:
            if not enemy.alive:
                continue
            
            # Check if tower can target this enemy type
            if enemy.is_flying and not self.can_hit_air:
                continue  # Can't hit flying enemies
            
            dx = enemy.x - self.x
            dy = enemy.y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance <= self.range and distance < min_distance:
                self.target = enemy
                min_distance = distance
        
        # Shoot if ready and has target
        if self.target and self.cooldown == 0:
            self.shoot()
            self.cooldown = self.max_cooldown
    
    def shoot(self):
        """Create a bullet towards the target."""
        bullet = Bullet(self.x, self.y, self.target, self.damage, self.tower_type)
        bullet_list.append(bullet)
    
    def draw(self):
        """Draw tower on screen."""
        rect_size = CELL_SIZE - 10
        rect_x = self.col * CELL_SIZE + 5
        rect_y = self.row * CELL_SIZE + 5
        pygame.draw.rect(screen, self.color, (rect_x, rect_y, rect_size, rect_size))
        
        # Tower top
        if self.tower_type == 'antiair':
            # Draw radar dish for anti-air tower
            pygame.draw.circle(screen, DARK_GREEN, (int(self.x), int(self.y)), 15)
            pygame.draw.circle(screen, CYAN, (int(self.x), int(self.y)), 12)
            # Radar lines
            for angle in range(0, 360, 45):
                end_x = self.x + math.cos(math.radians(angle)) * 8
                end_y = self.y + math.sin(math.radians(angle)) * 8
                pygame.draw.line(screen, DARK_CYAN, (int(self.x), int(self.y)), 
                               (int(end_x), int(end_y)), 2)
        else:
            pygame.draw.circle(screen, DARK_GREEN, (int(self.x), int(self.y)), 15)
        
        # Draw level indicator
        level_text = font_tiny.render(str(self.level), True, WHITE)
        text_rect = level_text.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(level_text, text_rect)
        
        # Draw range circle when targeting
        if self.target:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.range), 1)

# --- STEP 11: BULLET CLASS (ENHANCED) ---
class Bullet:
    """
    Bullet that moves toward enemy and deals damage on hit.
    """
    def __init__(self, x, y, target, damage, tower_type='basic'):
        self.x = x
        self.y = y
        self.target = target
        self.damage = damage
        self.tower_type = tower_type
        self.speed = BULLET_SPEED if tower_type != 'sniper' else BULLET_SPEED * 2
        self.radius = 5 if tower_type != 'rapid' else 3
        self.active = True
        
        # Different bullet colors
        if tower_type == 'sniper':
            self.color = PURPLE
        elif tower_type == 'rapid':
            self.color = ORANGE
        elif tower_type == 'antiair':
            self.color = CYAN
        else:
            self.color = BLACK
    
    def update(self):
        """Move bullet towards target."""
        if not self.target.alive:
            self.active = False
            return
        
        # Target position (adjust for flying enemies)
        target_y = self.target.y
        if self.target.is_flying:
            target_y += math.sin(self.target.hover_offset) * 5
        
        dx = self.target.x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.speed:
            self.target.take_damage(self.damage)
            self.active = False
            # Create impact particles
            for _ in range(3):
                particle_list.append(Particle(self.x, self.y, self.color))
        else:
            self.x += (dx / distance) * self.speed
            self.y += (dy / distance) * self.speed
    
    def draw(self):
        """Draw bullet on screen."""
        if self.tower_type == 'antiair':
            # Draw missile for anti-air
            pygame.draw.circle(screen, CYAN, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius - 2)
        else:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# --- STEP 12: HELPER FUNCTIONS ---
def spawn_enemy():
    """Create new enemy with wave difficulty scaling."""
    global enemies_in_wave, total_enemies_spawned
    
    if len(enemy_path) == 0:
        return
    
    health_mult = 1 + (wave_number - 1) * 0.3
    
    # Boss wave every 5 waves
    if wave_number % 5 == 0 and enemies_in_wave == 0:
        enemy = Enemy(enemy_path, health_mult, 'boss')
    # Flying enemies appear after wave 6
    elif wave_number >= 6 and random.random() < 0.3:  # 30% chance
        enemy = Enemy(None, health_mult, 'flying')
    # Mix of enemy types
    elif wave_number > 3:
        enemy_type = random.choice(['normal', 'fast', 'tank', 'normal'])
        enemy = Enemy(enemy_path, health_mult, enemy_type)
    else:
        enemy = Enemy(enemy_path, health_mult, 'normal')
    
    enemy_list.append(enemy)
    enemies_in_wave += 1
    total_enemies_spawned += 1

def place_tower(row, col):
    """Place tower at grid position if valid."""
    global money, enemy_path
    
    if row < 0 or row >= GRID_ROWS or col < 0 or col >= GRID_COLS:
        return False
    
    if grid[row][col] != 0:
        return False
    
    tower_cost = TOWER_TYPES[selected_tower_type]['cost']
    if money < tower_cost:
        return False
    
    grid[row][col] = 1
    
    new_path = calculate_path(spawn_row, spawn_col, base_row, base_col, grid)
    
    if len(new_path) == 0:
        grid[row][col] = 0
        return False
    
    money -= tower_cost
    tower = Tower(row, col, selected_tower_type)
    tower_list.append(tower)
    stats['towers_placed'] += 1
    
    enemy_path = new_path
    for enemy in enemy_list:
        if enemy.alive and not enemy.is_flying and enemy.path_index < len(enemy.path) - 1:
            enemy.path = new_path[:]
            min_dist = float('inf')
            closest_idx = 0
            for i, (r, c) in enumerate(new_path):
                target_x = c * CELL_SIZE + CELL_SIZE // 2
                target_y = r * CELL_SIZE + CELL_SIZE // 2
                dist = math.sqrt((enemy.x - target_x)**2 + (enemy.y - target_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            enemy.path_index = max(0, closest_idx - 1)
    
    return True

def sell_tower(row, col):
    """Sell tower and get 70% refund."""
    global money
    
    for tower in tower_list:
        if tower.row == row and tower.col == col:
            refund = int(TOWER_TYPES[tower.tower_type]['cost'] * 0.7)
            money += refund
            tower_list.remove(tower)
            grid[row][col] = 0
            
            global enemy_path
            enemy_path = calculate_path(spawn_row, spawn_col, base_row, base_col, grid)
            return True
    return False

def upgrade_tower(row, col):
    """Upgrade tower if possible."""
    global money
    
    for tower in tower_list:
        if tower.row == row and tower.col == col:
            if money >= tower.upgrade_cost:
                money -= tower.upgrade_cost
                tower.upgrade()
                return True
    return False

def get_grid_pos(mouse_x, mouse_y):
    """Convert mouse position to grid coordinates."""
    col = mouse_x // CELL_SIZE
    row = mouse_y // CELL_SIZE
    return row, col

def draw_path():
    """Draw the enemy path on the grid."""
    for i, (row, col) in enumerate(enemy_path):
        x = col * CELL_SIZE + CELL_SIZE // 2
        y = row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, YELLOW, (x, y), 3)

def draw_grid():
    """Draw the game grid."""
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            
            if grid[row][col] == 0:
                pygame.draw.rect(screen, LIGHT_GRAY, (x, y, CELL_SIZE, CELL_SIZE))
            
            pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)
    
    # Draw base
    base_x = base_col * CELL_SIZE
    base_y = base_row * CELL_SIZE
    pygame.draw.rect(screen, BLUE, (base_x + 5, base_y + 5, CELL_SIZE - 10, CELL_SIZE - 10))
    pygame.draw.circle(screen, YELLOW, (base_x + CELL_SIZE // 2, base_y + CELL_SIZE // 2), 10)

def draw_tower_preview(mouse_x, mouse_y):
    """Draw preview of tower placement."""
    row, col = get_grid_pos(mouse_x, mouse_y)
    
    if row < 0 or row >= GRID_ROWS or col < 0 or col >= GRID_COLS:
        return
    
    if grid[row][col] != 0:
        return
    
    tower_cost = TOWER_TYPES[selected_tower_type]['cost']
    if money < tower_cost:
        return
    
    x = col * CELL_SIZE + CELL_SIZE // 2
    y = row * CELL_SIZE + CELL_SIZE // 2
    
    tower_range = TOWER_TYPES[selected_tower_type]['range']
    pygame.draw.circle(screen, GREEN, (x, y), tower_range, 1)
    
    s = pygame.Surface((CELL_SIZE - 10, CELL_SIZE - 10))
    s.set_alpha(128)
    s.fill(TOWER_TYPES[selected_tower_type]['color'])
    screen.blit(s, (col * CELL_SIZE + 5, row * CELL_SIZE + 5))

def draw_tower_buttons():
    """Draw tower selection buttons."""
    button_y = GRID_ROWS * CELL_SIZE + 50
    button_x = 10
    button_width = 70
    button_height = 40
    spacing = 80
    
    for i, (tower_type, stats) in enumerate(TOWER_TYPES.items()):
        x = button_x + i * spacing
        
        # Button background
        color = stats['color'] if selected_tower_type != tower_type else YELLOW
        pygame.draw.rect(screen, color, (x, button_y, button_width, button_height))
        pygame.draw.rect(screen, BLACK, (x, button_y, button_width, button_height), 2)
        
        # Tower name (shortened)
        if tower_type == 'antiair':
            name = 'AA'
        else:
            name = stats['name'].split()[0][:6]
        name_text = font_tiny.render(name, True, BLACK)
        screen.blit(name_text, (x + 5, button_y + 5))
        
        # Cost
        cost_text = font_tiny.render(f"${stats['cost']}", True, BLACK)
        screen.blit(cost_text, (x + 5, button_y + 22))

def draw_ui():
    """Draw user interface."""
    ui_y = GRID_ROWS * CELL_SIZE
    pygame.draw.rect(screen, GRAY, (0, ui_y, WINDOW_WIDTH, 150))
    
    # Top row stats
    health_text = font_medium.render(f"Base: {base_health}/{max_base_health}", True, BLACK)
    screen.blit(health_text, (10, ui_y + 10))
    
    money_color = BLACK
    money_text = font_medium.render(f"Money: ${money}", True, money_color)
    screen.blit(money_text, (220, ui_y + 10))
    
    score_text = font_medium.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (420, ui_y + 10))
    
    wave_text = font_medium.render(f"Wave: {wave_number}", True, BLACK)
    screen.blit(wave_text, (580, ui_y + 10))
    
    # Draw tower selection buttons
    draw_tower_buttons()
    
    # Instructions
    inst_text = font_small.render("L-Click: Place | R-Click: Sell | P: Pause", True, BLACK)
    screen.blit(inst_text, (320, ui_y + 100))
    
    # Wave countdown
    if enemies_in_wave >= WAVE_SIZE and len(enemy_list) == 0:
        countdown = (WAVE_DELAY - wave_timer) // 60
        countdown_text = font_medium.render(f"Next wave in: {countdown}s", True, BLUE)
        screen.blit(countdown_text, (300, ui_y + 120))
    
    # Flying enemy warning
    if wave_number >= 6:
        warning = font_tiny.render("⚠️ Flying enemies active! Use Anti-Air towers", True, RED)
        screen.blit(warning, (10, ui_y + 120))

def draw_base_health_bar():
    """Draw visual health bar for base with better styling."""
    bar_width = 250
    bar_height = 25
    bar_x = WINDOW_WIDTH - bar_width - 15
    bar_y = 15
    
    health_percent = max(0, base_health / max_base_health)
    
    if health_percent > 0.6:
        bar_color = GREEN
    elif health_percent > 0.3:
        bar_color = YELLOW
    else:
        bar_color = RED
    
    pygame.draw.rect(screen, BLACK, (bar_x - 2, bar_y - 2, bar_width + 4, bar_height + 4))
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(screen, bar_color, (bar_x, bar_y, int(bar_width * health_percent), bar_height))
    pygame.draw.rect(screen, WHITE, (bar_x, bar_y, bar_width, bar_height), 2)
    
    health_text = font_small.render(f"Base: {base_health}/{max_base_health}", True, WHITE)
    text_rect = health_text.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
    shadow_rect = health_text.get_rect(center=(bar_x + bar_width // 2 + 1, bar_y + bar_height // 2 + 1))
    shadow_text = font_small.render(f"Base: {base_health}/{max_base_health}", True, BLACK)
    screen.blit(shadow_text, shadow_rect)
    screen.blit(health_text, text_rect)
    
    label_text = font_tiny.render("🏰 BASE HEALTH", True, BLACK)
    screen.blit(label_text, (bar_x, bar_y - 18))

def draw_pause_menu():
    """Draw pause menu overlay."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    text = font_large.render("PAUSED", True, YELLOW)
    text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
    screen.blit(text, text_rect)
    
    inst = font_medium.render("Press P to Resume", True, WHITE)
    inst_rect = inst.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
    screen.blit(inst, inst_rect)
    
    esc = font_small.render("Press ESC to Exit", True, WHITE)
    esc_rect = esc.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
    screen.blit(esc, esc_rect)

def draw_game_over():
    """Draw game over screen."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    if game_won:
        text = font_large.render("YOU WON!", True, YELLOW)
    else:
        text = font_large.render("GAME OVER", True, RED)
    
    text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))
    screen.blit(text, text_rect)
    
    # Statistics
    stats_y = WINDOW_HEIGHT // 2 - 20
    score_text = font_medium.render(f"Final Score: {score}", True, WHITE)
    score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y))
    screen.blit(score_text, score_rect)
    
    wave_text = font_medium.render(f"Wave Reached: {wave_number}", True, WHITE)
    wave_rect = wave_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 35))
    screen.blit(wave_text, wave_rect)
    
    kills_text = font_small.render(f"Total Kills: {stats['total_kills']}", True, WHITE)
    kills_rect = kills_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 65))
    screen.blit(kills_text, kills_rect)
    
    towers_text = font_small.render(f"Towers Placed: {stats['towers_placed']}", True, WHITE)
    towers_rect = towers_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 90))
    screen.blit(towers_text, towers_rect)
    
    restart_text = font_small.render("Press ESC to exit", True, WHITE)
    restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 130))
    screen.blit(restart_text, restart_rect)

# --- STEP 13: MAIN GAME LOOP ---
running = True
while running:
    clock.tick(FPS)
    
    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_p:
                paused = not paused
        
        # Mouse click
        if event.type == pygame.MOUSEBUTTONDOWN and not game_over and not paused:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            # Check if clicking on tower selection buttons
            if mouse_y >= GRID_ROWS * CELL_SIZE + 50 and mouse_y <= GRID_ROWS * CELL_SIZE + 90:
                for i, tower_type in enumerate(TOWER_TYPES.keys()):
                    button_x = 10 + i * 80
                    if mouse_x >= button_x and mouse_x <= button_x + 70:
                        selected_tower_type = tower_type
                        break
            
            # Clicking on grid
            elif mouse_y < GRID_ROWS * CELL_SIZE:
                row, col = get_grid_pos(mouse_x, mouse_y)
                
                if event.button == 1:  # Left click - place tower
                    place_tower(row, col)
                elif event.button == 3:  # Right click - sell tower
                    sell_tower(row, col)
    
    # --- GAME LOGIC (only if not game over or paused) ---
    if not game_over and not paused:
        # Spawn enemies in waves
        spawn_timer += 1
        if enemies_in_wave < WAVE_SIZE and spawn_timer >= ENEMY_SPAWN_INTERVAL:
            spawn_timer = 0
            spawn_enemy()
        
        # Check if wave is complete
        if enemies_in_wave >= WAVE_SIZE and len(enemy_list) == 0:
            wave_timer += 1
            if wave_timer >= WAVE_DELAY:
                wave_number += 1
                enemies_in_wave = 0
                wave_timer = 0
                money += 50
                stats['highest_wave'] = wave_number
        
        # Update enemies
        enemies_to_remove = []
        for enemy in enemy_list:
            if not enemy.alive:
                enemies_to_remove.append(enemy)
                money += enemy.reward
                score += 10
                stats['total_kills'] += 1
                stats['money_earned'] += enemy.reward
                if enemy.is_flying:
                    stats['flying_kills'] += 1
                continue
            
            enemy.move()
            
            if enemy.reached_base():
                base_health -= 10
                enemies_to_remove.append(enemy)
        
        for enemy in enemies_to_remove:
            enemy_list.remove(enemy)
        
        # Update towers
        for tower in tower_list:
            tower.update(enemy_list)
        
        # Update bullets
        bullets_to_remove = []
        for bullet in bullet_list:
            if not bullet.active:
                bullets_to_remove.append(bullet)
                continue
            bullet.update()
        
        for bullet in bullets_to_remove:
            bullet_list.remove(bullet)
        
        # Update particles
        particles_to_remove = []
        for particle in particle_list:
            particle.update()
            if particle.life <= 0:
                particles_to_remove.append(particle)
        
        for particle in particles_to_remove:
            particle_list.remove(particle)
        
        # Check game over
        if base_health <= 0:
            game_over = True
            game_won = False
        
        # Check win condition
        if wave_number > 15:
            game_over = True
            game_won = True
    
    # --- DRAWING ---
    screen.fill(WHITE)
    
    draw_grid()
    draw_path()
    
    for tower in tower_list:
        tower.draw()
    
    for enemy in enemy_list:
        enemy.draw()
    
    for bullet in bullet_list:
        bullet.draw()
    
    for particle in particle_list:
        particle.draw()
    
    draw_base_health_bar()
    
    if not game_over and not paused:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_y < GRID_ROWS * CELL_SIZE:
            draw_tower_preview(mouse_x, mouse_y)
    
    draw_ui()
    
    if paused:
        draw_pause_menu()
    
    if game_over:
        draw_game_over()
    
    pygame.display.flip()

# --- CLEANUP ---
pygame.quit()
sys.exit()