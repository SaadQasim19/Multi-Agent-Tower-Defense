# =============================================================================
# 🏰 ADVANCED TOWER DEFENSE GAME - Multi-Agent AI System with CrewAI
# =============================================================================

# --- STEP 1: IMPORT LIBRARIES ---
import pygame
import sys
import math
import random
from collections import deque
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables FIRST
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Verify the API key is loaded
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("ERROR: GROQ_API_KEY not found in .env file!")
    print(f"Looking for .env at: {env_path}")
    sys.exit(1)
else:
    print(f"✓ GROQ API Key loaded successfully (starts with: {groq_api_key[:10]}...)")

# Set environment variables for CrewAI
os.environ["OPENAI_API_KEY"] = groq_api_key
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["CREWAI_TRACING_ENABLED"] = "false"  # Disable tracing prompts

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# --- STEP 2: INITIALIZE PYGAME ---
pygame.init()

# --- STEP 3: GAME SETTINGS ---
GRID_COLS = 15
GRID_ROWS = 10
CELL_SIZE = 50
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 200

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
DARK_BLUE = (0, 0, 139)

# Game settings
FPS = 60
ENEMY_SPEED = 1.5
BULLET_SPEED = 5

# Tower types
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
ENEMY_SPAWN_INTERVAL = 90
WAVE_SIZE = 5
WAVE_DELAY = 180

# --- STEP 4: CREATE GAME WINDOW ---
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("🏰 Tower Defense AI Commander - GROQ Powered")
clock = pygame.time.Clock()

# Fonts
font_large = pygame.font.Font(None, 48)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)
font_tiny = pygame.font.Font(None, 18)

# --- STEP 5: GAME STATE ---
game_state = 'menu'

# --- STEP 6: CREATE GRID SYSTEM ---
def init_grid():
    grid = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    grid[GRID_ROWS - 1][GRID_COLS - 1] = 2
    return grid

grid = init_grid()

base_row = GRID_ROWS - 1
base_col = GRID_COLS - 1
spawn_row = 0
spawn_col = 0

# --- STEP 7: BFS PATHFINDING ---
def calculate_path(start_row, start_col, goal_row, goal_col, current_grid):
    rows = len(current_grid)
    cols = len(current_grid[0])
    
    queue = deque()
    queue.append((start_row, start_col, [(start_row, start_col)]))
    
    visited = set()
    visited.add((start_row, start_col))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        current_row, current_col, path = queue.popleft()
        
        if current_row == goal_row and current_col == goal_col:
            return path
        
        for dr, dc in directions:
            new_row = current_row + dr
            new_col = current_col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if (new_row, new_col) not in visited:
                    if current_grid[new_row][new_col] != 1:
                        visited.add((new_row, new_col))
                        new_path = path + [(new_row, new_col)]
                        queue.append((new_row, new_col, new_path))
    
    return []

# --- STEP 8: GAME VARIABLES ---
def init_game_variables():
    global base_health, max_base_health, money, score, game_over, game_won
    global wave_number, enemies_in_wave, total_enemies_spawned, paused
    global selected_tower_type, selected_tower
    global enemy_list, tower_list, bullet_list, particle_list
    global spawn_timer, wave_timer, stats, enemy_path, grid
    global ai_advice, ai_advice_timer, ai_requesting, ai_pause_mode
    
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
    
    enemy_list = []
    tower_list = []
    bullet_list = []
    particle_list = []
    
    spawn_timer = 0
    wave_timer = 0
    
    stats = {
        'total_kills': 0,
        'towers_placed': 0,
        'money_earned': 0,
        'highest_wave': 1,
        'flying_kills': 0
    }
    
    # AI Commander variables
    ai_advice = []  # List of advice lines
    ai_advice_timer = 0
    ai_requesting = False
    ai_pause_mode = False  # New: Special pause for AI advice
    
    grid = init_grid()
    enemy_path = calculate_path(spawn_row, spawn_col, base_row, base_col, grid)

init_game_variables()

# --- STEP 9: CrewAI SETUP ---
llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

strategy_advisor = Agent(
    role='Tower Defense Strategy Commander',
    goal='Provide tactical advice to help player defend their base efficiently',
    backstory="""You are a friendly military strategist specializing in tower defense.
    You analyze battlefield conditions and provide helpful, concise advice in a warm and 
    encouraging tone. Keep responses brief and actionable.""",
    verbose=False,
    allow_delegation=False,
    llm=llm
)

def get_game_state_summary():
    tower_counts = {'basic': 0, 'sniper': 0, 'rapid': 0, 'antiair': 0}
    for tower in tower_list:
        tower_counts[tower.tower_type] += 1
    
    enemy_counts = {'normal': 0, 'fast': 0, 'tank': 0, 'boss': 0, 'flying': 0}
    for enemy in enemy_list:
        enemy_counts[enemy.enemy_type] += 1
    
    return {
        'wave': wave_number,
        'base_health': base_health,
        'money': money,
        'score': score,
        'towers': tower_counts,
        'total_towers': len(tower_list),
        'active_enemies': len(enemy_list),
        'enemy_types': enemy_counts,
        'total_kills': stats['total_kills'],
        'flying_kills': stats['flying_kills']
    }

def get_ai_advice():
    global ai_advice, ai_advice_timer, ai_requesting, ai_pause_mode
    
    try:
        game_state = get_game_state_summary()
        
        analysis_task = Task(
            description=f"""Analyze this tower defense battle and give 2-3 tactical tips:
            
            📊 BATTLEFIELD STATUS:
            Wave: {game_state['wave']} | Base Health: {game_state['base_health']}/100 | Money: ${game_state['money']}
            Active Enemies: {game_state['active_enemies']} (Flying: {game_state['enemy_types']['flying']})
            Towers: Basic:{game_state['towers']['basic']}, Sniper:{game_state['towers']['sniper']}, 
                    Rapid:{game_state['towers']['rapid']}, Anti-Air:{game_state['towers']['antiair']}
            
            Give 2-3 friendly, brief tips. Each tip should be one short sentence.
            Format each tip on a new line starting with "•"
            Example:
            • Build more towers near the path!
            • Get Anti-Air towers for flying enemies
            • Save money for the next wave
            """,
            agent=strategy_advisor,
            expected_output="2-3 brief tactical tips, each on a new line starting with bullet point"
        )
        
        crew = Crew(
            agents=[strategy_advisor],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        advice_text = str(result).strip()
        
        # Parse advice into lines
        lines = []
        for line in advice_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                # Clean up bullet point
                if not line.startswith('•'):
                    line = '• ' + line[1:].strip()
                lines.append(line)
        
        if not lines:
            lines = ["• Stay focused and keep defending!"]
        
        ai_advice = ["🎖️ AI COMMANDER ADVICE:", ""] + lines
        ai_advice_timer = 600  # Show for 10 seconds
        
    except Exception as e:
        print(f"AI Error: {e}")
        game_state = get_game_state_summary()
        
        advice_lines = ["🎖️ AI COMMANDER ADVICE:", ""]
        
        if game_state['base_health'] < 30:
            advice_lines.append("• ⚠️ Critical! Strengthen defenses now!")
        elif game_state['money'] > 200:
            advice_lines.append("• 💰 You have resources - build more towers!")
        
        if game_state['enemy_types']['flying'] > 0 and game_state['towers']['antiair'] == 0:
            advice_lines.append("• ✈️ Get Anti-Air towers for flying enemies!")
        elif wave_number >= 5 and game_state['total_towers'] < 5:
            advice_lines.append("• 🏰 You need more towers for this wave!")
        else:
            advice_lines.append("• 👍 Good work! Keep it up!")
        
        ai_advice = advice_lines
        ai_advice_timer = 600
    
    finally:
        ai_requesting = False
        ai_pause_mode = True  # Enter AI pause mode

# --- STEP 10: PARTICLE CLASS ---
class Particle:
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
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.vy += 0.2
    
    def draw(self):
        alpha = int(255 * (self.life / self.max_life))
        size = int(self.size * (self.life / self.max_life))
        if size > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), size)

# --- STEP 11: ENEMY CLASS ---
class Enemy:
    def __init__(self, path, health_multiplier=1, enemy_type='normal'):
        self.enemy_type = enemy_type
        self.is_flying = (enemy_type == 'flying')
        
        if self.is_flying:
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
        elif enemy_type == 'flying':
            self.health = 40 * health_multiplier
            self.max_health = 40 * health_multiplier
            self.speed = ENEMY_SPEED * 1.3
            self.color = LIGHT_BLUE
            self.reward = 35
            self.radius = 11
            self.hover_offset = 0
        else:
            self.health = 50 * health_multiplier
            self.max_health = 50 * health_multiplier
            self.speed = ENEMY_SPEED
            self.color = RED
            self.reward = 25
            self.radius = 12
        
        self.alive = True
    
    def move(self):
        if self.is_flying:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > self.speed:
                self.x += (dx / distance) * self.speed
                self.y += (dy / distance) * self.speed
            else:
                self.x = self.target_x
                self.y = self.target_y
            
            self.hover_offset = (self.hover_offset + 0.1) % (2 * math.pi)
        else:
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
        self.health -= damage
        if self.health <= 0:
            self.alive = False
            for _ in range(10):
                particle_list.append(Particle(self.x, self.y, self.color))
    
    def reached_base(self):
        if self.is_flying:
            distance = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
            return distance < 10
        else:
            return self.path_index >= len(self.path) - 1
    
    def draw(self):
        draw_y = self.y
        if self.is_flying:
            draw_y = self.y + math.sin(self.hover_offset) * 5
        
        if self.is_flying:
            pygame.draw.ellipse(screen, GRAY, 
                              (int(self.x - self.radius), 
                               int(self.y + self.radius), 
                               self.radius * 2, self.radius))
        
        pygame.draw.circle(screen, self.color, (int(self.x), int(draw_y)), self.radius)
        
        if self.is_flying:
            wing_offset = abs(math.sin(self.hover_offset * 2)) * 5
            wing_left = [(int(self.x - self.radius - wing_offset), int(draw_y)),
                        (int(self.x - self.radius - 5 - wing_offset), int(draw_y - 8)),
                        (int(self.x - self.radius), int(draw_y - 3))]
            pygame.draw.polygon(screen, DARK_CYAN, wing_left)
            
            wing_right = [(int(self.x + self.radius + wing_offset), int(draw_y)),
                         (int(self.x + self.radius + 5 + wing_offset), int(draw_y - 8)),
                         (int(self.x + self.radius), int(draw_y - 3))]
            pygame.draw.polygon(screen, DARK_CYAN, wing_right)
        
        bar_width = max(30, self.radius * 2)
        bar_height = 6
        health_percent = max(0, min(1, self.health / self.max_health))
        
        bar_x = self.x - bar_width // 2
        bar_y = draw_y - self.radius - 15
        
        pygame.draw.rect(screen, BLACK, (bar_x - 1, bar_y - 1, bar_width + 2, bar_height + 2))
        pygame.draw.rect(screen, DARK_RED, (bar_x, bar_y, bar_width, bar_height))
        
        if health_percent > 0.6:
            health_color = GREEN
        elif health_percent > 0.3:
            health_color = YELLOW
        else:
            health_color = ORANGE
        
        pygame.draw.rect(screen, health_color, (bar_x, bar_y, int(bar_width * health_percent), bar_height))
        
        if self.enemy_type == 'boss':
            pygame.draw.circle(screen, YELLOW, (int(self.x), int(draw_y)), self.radius + 3, 2)
            health_text = font_tiny.render(f"{int(self.health)}", True, WHITE)
            text_rect = health_text.get_rect(center=(int(self.x), bar_y - 10))
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                outline_rect = health_text.get_rect(center=(int(self.x) + dx, bar_y - 10 + dy))
                screen.blit(health_text, outline_rect)
            screen.blit(health_text, text_rect)
        elif self.is_flying:
            pygame.draw.circle(screen, WHITE, (int(self.x), int(draw_y)), self.radius + 2, 1)

# --- STEP 12: TOWER CLASS ---
class Tower:
    def __init__(self, row, col, tower_type='basic'):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE + CELL_SIZE // 2
        self.y = row * CELL_SIZE + CELL_SIZE // 2
        self.tower_type = tower_type
        self.level = 1
        self.load_stats()
        self.cooldown = 0
        self.target = None
    
    def load_stats(self):
        stats = TOWER_TYPES[self.tower_type]
        self.range = stats['range'] * (1 + (self.level - 1) * 0.2)
        self.damage = stats['damage'] * (1 + (self.level - 1) * 0.5)
        self.max_cooldown = stats['cooldown']
        self.color = stats['color']
        self.upgrade_cost = stats['cost'] * self.level
        self.can_hit_air = stats['can_hit_air']
    
    def upgrade(self):
        self.level += 1
        self.load_stats()
    
    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
        
        self.target = None
        min_distance = self.range + 1
        
        for enemy in enemies:
            if not enemy.alive:
                continue
            
            if enemy.is_flying and not self.can_hit_air:
                continue
            
            dx = enemy.x - self.x
            dy = enemy.y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance <= self.range and distance < min_distance:
                self.target = enemy
                min_distance = distance
        
        if self.target and self.cooldown == 0:
            self.shoot()
            self.cooldown = self.max_cooldown
    
    def shoot(self):
        bullet = Bullet(self.x, self.y, self.target, self.damage, self.tower_type)
        bullet_list.append(bullet)
    
    def draw(self):
        rect_size = CELL_SIZE - 10
        rect_x = self.col * CELL_SIZE + 5
        rect_y = self.row * CELL_SIZE + 5
        pygame.draw.rect(screen, self.color, (rect_x, rect_y, rect_size, rect_size))
        
        if self.tower_type == 'antiair':
            pygame.draw.circle(screen, DARK_GREEN, (int(self.x), int(self.y)), 15)
            pygame.draw.circle(screen, CYAN, (int(self.x), int(self.y)), 12)
            for angle in range(0, 360, 45):
                end_x = self.x + math.cos(math.radians(angle)) * 8
                end_y = self.y + math.sin(math.radians(angle)) * 8
                pygame.draw.line(screen, DARK_CYAN, (int(self.x), int(self.y)), 
                               (int(end_x), int(end_y)), 2)
        else:
            pygame.draw.circle(screen, DARK_GREEN, (int(self.x), int(self.y)), 15)
        
        level_text = font_tiny.render(str(self.level), True, WHITE)
        text_rect = level_text.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(level_text, text_rect)
        
        if self.target:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.range), 1)

# --- STEP 13: BULLET CLASS ---
class Bullet:
    def __init__(self, x, y, target, damage, tower_type='basic'):
        self.x = x
        self.y = y
        self.target = target
        self.damage = damage
        self.tower_type = tower_type
        self.speed = BULLET_SPEED if tower_type != 'sniper' else BULLET_SPEED * 2
        self.radius = 5 if tower_type != 'rapid' else 3
        self.active = True
        
        if tower_type == 'sniper':
            self.color = PURPLE
        elif tower_type == 'rapid':
            self.color = ORANGE
        elif tower_type == 'antiair':
            self.color = CYAN
        else:
            self.color = BLACK
    
    def update(self):
        if not self.target.alive:
            self.active = False
            return
        
        target_y = self.target.y
        if self.target.is_flying:
            target_y += math.sin(self.target.hover_offset) * 5
        
        dx = self.target.x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.speed:
            self.target.take_damage(self.damage)
            self.active = False
            for _ in range(3):
                particle_list.append(Particle(self.x, self.y, self.color))
        else:
            self.x += (dx / distance) * self.speed
            self.y += (dy / distance) * self.speed
    
    def draw(self):
        if self.tower_type == 'antiair':
            pygame.draw.circle(screen, CYAN, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius - 2)
        else:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

# --- STEP 14: HELPER FUNCTIONS ---
def spawn_enemy():
    global enemies_in_wave, total_enemies_spawned
    
    if len(enemy_path) == 0:
        return
    
    health_mult = 1 + (wave_number - 1) * 0.3
    
    if wave_number % 5 == 0 and enemies_in_wave == 0:
        enemy = Enemy(enemy_path, health_mult, 'boss')
    elif wave_number >= 6 and random.random() < 0.3:
        enemy = Enemy(None, health_mult, 'flying')
    elif wave_number > 3:
        enemy_type = random.choice(['normal', 'fast', 'tank', 'normal'])
        enemy = Enemy(enemy_path, health_mult, enemy_type)
    else:
        enemy = Enemy(enemy_path, health_mult, 'normal')
    
    enemy_list.append(enemy)
    enemies_in_wave += 1
    total_enemies_spawned += 1

def place_tower(row, col):
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

def get_grid_pos(mouse_x, mouse_y):
    col = mouse_x // CELL_SIZE
    row = mouse_y // CELL_SIZE
    return row, col

def draw_path():
    for i, (row, col) in enumerate(enemy_path):
        x = col * CELL_SIZE + CELL_SIZE // 2
        y = row * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, YELLOW, (x, y), 3)

def draw_grid():
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            
            if grid[row][col] == 0:
                pygame.draw.rect(screen, LIGHT_GRAY, (x, y, CELL_SIZE, CELL_SIZE))
            
            pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)
    
    base_x = base_col * CELL_SIZE
    base_y = base_row * CELL_SIZE
    pygame.draw.rect(screen, BLUE, (base_x + 5, base_y + 5, CELL_SIZE - 10, CELL_SIZE - 10))
    pygame.draw.circle(screen, YELLOW, (base_x + CELL_SIZE // 2, base_y + CELL_SIZE // 2), 10)

def draw_tower_preview(mouse_x, mouse_y):
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
    button_y = GRID_ROWS * CELL_SIZE + 50
    button_x = 10
    button_width = 70
    button_height = 40
    spacing = 80
    
    for i, (tower_type, stats) in enumerate(TOWER_TYPES.items()):
        x = button_x + i * spacing
        
        color = stats['color'] if selected_tower_type != tower_type else YELLOW
        pygame.draw.rect(screen, color, (x, button_y, button_width, button_height))
        pygame.draw.rect(screen, BLACK, (x, button_y, button_width, button_height), 2)
        
        if tower_type == 'antiair':
            name = 'AA'
        else:
            name = stats['name'].split()[0][:6]
        name_text = font_tiny.render(name, True, BLACK)
        screen.blit(name_text, (x + 5, button_y + 5))
        
        cost_text = font_tiny.render(f"${stats['cost']}", True, BLACK)
        screen.blit(cost_text, (x + 5, button_y + 22))

def draw_ai_commander_panel():
    """Draw AI Commander panel with advice or request button"""
    panel_y = GRID_ROWS * CELL_SIZE + 100
    panel_height = 90
    
    # Dark background
    pygame.draw.rect(screen, (20, 20, 40), (0, panel_y, WINDOW_WIDTH, panel_height))
    pygame.draw.rect(screen, CYAN, (0, panel_y, WINDOW_WIDTH, panel_height), 3)
    
    # AI Icon
    pygame.draw.circle(screen, CYAN, (30, panel_y + 30), 18)
    pygame.draw.circle(screen, (20, 20, 40), (30, panel_y + 30), 14)
    ai_icon_text = font_small.render("AI", True, CYAN)
    screen.blit(ai_icon_text, (18, panel_y + 20))
    
    if ai_requesting:
        # Show loading animation
        loading_text = font_medium.render("🤔 Analyzing battlefield...", True, YELLOW)
        screen.blit(loading_text, (70, panel_y + 15))
    elif ai_advice_timer > 0:
        # Show advice
        y_offset = panel_y + 10
        for line in ai_advice:
            if line.startswith("🎖️"):
                text = font_small.render(line, True, YELLOW)
            else:
                text = font_tiny.render(line, True, WHITE)
            screen.blit(text, (70, y_offset))
            y_offset += 22
    else:
        # Show request button
        hint_text = font_small.render("Press 'A' for AI Tactical Advice", True, LIGHT_GRAY)
        screen.blit(hint_text, (70, panel_y + 20))
        
        button_rect = pygame.Rect(WINDOW_WIDTH - 150, panel_y + 15, 130, 35)
        pygame.draw.rect(screen, CYAN, button_rect)
        pygame.draw.rect(screen, WHITE, button_rect, 2)
        button_text = font_small.render("Get Advice", True, BLACK)
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)

def draw_ai_pause_overlay():
    """Draw special overlay when AI advice is shown"""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 50))
    screen.blit(overlay, (0, 0))
    
    # Title
    title = font_large.render("🎖️ AI COMMANDER", True, CYAN)
    title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
    screen.blit(title, title_rect)
    
    # Advice box
    box_width = 600
    box_height = 300
    box_x = (WINDOW_WIDTH - box_width) // 2
    box_y = 220
    
    pygame.draw.rect(screen, (30, 30, 60), (box_x, box_y, box_width, box_height))
    pygame.draw.rect(screen, CYAN, (box_x, box_y, box_width, box_height), 3)
    
    # Display advice
    y_offset = box_y + 30
    for line in ai_advice:
        if line.startswith("🎖️"):
            text = font_medium.render(line, True, YELLOW)
        elif line == "":
            y_offset += 10
            continue
        else:
            text = font_small.render(line, True, WHITE)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y_offset))
        screen.blit(text, text_rect)
        y_offset += 35
    
    # Continue button
    button_width = 200
    button_height = 50
    button_x = (WINDOW_WIDTH - button_width) // 2
    button_y = box_y + box_height + 30
    
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    pygame.draw.rect(screen, GREEN, button_rect)
    pygame.draw.rect(screen, WHITE, button_rect, 3)
    
    button_text = font_medium.render("CONTINUE", True, BLACK)
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)
    
    # Hint
    hint = font_tiny.render("Press SPACE or click CONTINUE", True, LIGHT_GRAY)
    hint_rect = hint.get_rect(center=(WINDOW_WIDTH // 2, button_y + button_height + 20))
    screen.blit(hint, hint_rect)
    
    return button_rect

def draw_ui():
    ui_y = GRID_ROWS * CELL_SIZE
    pygame.draw.rect(screen, GRAY, (0, ui_y, WINDOW_WIDTH, 200))
    
    health_text = font_medium.render(f"Base: {base_health}/{max_base_health}", True, BLACK)
    screen.blit(health_text, (10, ui_y + 10))
    
    money_text = font_medium.render(f"Money: ${money}", True, BLACK)
    screen.blit(money_text, (220, ui_y + 10))
    
    score_text = font_medium.render(f"Score: {score}", True, BLACK)
    screen.blit(score_text, (420, ui_y + 10))
    
    wave_text = font_medium.render(f"Wave: {wave_number}", True, BLACK)
    screen.blit(wave_text, (580, ui_y + 10))
    
    draw_tower_buttons()
    draw_ai_commander_panel()
    
    if enemies_in_wave >= WAVE_SIZE and len(enemy_list) == 0:
        countdown = (WAVE_DELAY - wave_timer) // 60
        countdown_text = font_small.render(f"Next wave: {countdown}s", True, BLUE)
        screen.blit(countdown_text, (10, ui_y + 160))
    
    if wave_number >= 6:
        warning = font_tiny.render("⚠️ Flying enemies! Use AA", True, RED)
        screen.blit(warning, (200, ui_y + 160))

def draw_base_health_bar():
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
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    text = font_large.render("⏸️ PAUSED", True, YELLOW)
    text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
    screen.blit(text, text_rect)
    
    inst = font_medium.render("Press P to Resume", True, WHITE)
    inst_rect = inst.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
    screen.blit(inst, inst_rect)
    
    esc = font_small.render("Press ESC to Menu", True, WHITE)
    esc_rect = esc.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
    screen.blit(esc, esc_rect)

def draw_main_menu():
    screen.fill(BLACK)
    
    title = font_large.render("🏰 TOWER DEFENSE AI 🏰", True, YELLOW)
    title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 120))
    for offset in [2, 4, 6]:
        glow = font_large.render("🏰 TOWER DEFENSE AI 🏰", True, ORANGE)
        glow_rect = glow.get_rect(center=(WINDOW_WIDTH // 2, 120))
        glow_rect.inflate_ip(offset, offset)
        screen.blit(glow, glow_rect)
    screen.blit(title, title_rect)
    
    subtitle = font_small.render("with GROQ-Powered AI Commander", True, CYAN)
    subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 170))
    screen.blit(subtitle, subtitle_rect)
    
    start_button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 250, 200, 50)
    pygame.draw.rect(screen, GREEN, start_button_rect)
    pygame.draw.rect(screen, WHITE, start_button_rect, 3)
    start_text = font_medium.render("START GAME", True, BLACK)
    start_text_rect = start_text.get_rect(center=start_button_rect.center)
    screen.blit(start_text, start_text_rect)
    
    instructions = [
        "HOW TO PLAY:",
        "• Click to place towers on the grid",
        "• Stop enemies from reaching your base",
        "• Press 'A' for AI Commander advice (pauses game)",
        "• Flying enemies need Anti-Air towers!",
        "",
        "Controls:",
        "• Left Click: Place Tower",
        "• Right Click: Sell Tower (70% refund)",
        "• A: AI Advice (pauses game)",
        "• P: Pause/Resume",
        "• ESC: Main Menu"
    ]
    
    y_offset = 340
    for line in instructions:
        if line.startswith("•"):
            text = font_tiny.render(line, True, LIGHT_GRAY)
        elif line.endswith(":"):
            text = font_small.render(line, True, YELLOW)
        else:
            text = font_tiny.render(line, True, WHITE)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y_offset))
        screen.blit(text, text_rect)
        y_offset += 25 if line.endswith(":") or line == "" else 20
    
    return start_button_rect

def draw_game_over():
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    if game_won:
        text = font_large.render("🎉 VICTORY! 🎉", True, YELLOW)
    else:
        text = font_large.render("💥 GAME OVER 💥", True, RED)
    
    text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 150))
    screen.blit(text, text_rect)
    
    stats_y = WINDOW_HEIGHT // 2 - 80
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
    
    flying_text = font_small.render(f"Flying Kills: {stats['flying_kills']}", True, CYAN)
    flying_rect = flying_text.get_rect(center=(WINDOW_WIDTH // 2, stats_y + 115))
    screen.blit(flying_text, flying_rect)
    
    restart_button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, stats_y + 160, 200, 50)
    pygame.draw.rect(screen, GREEN, restart_button_rect)
    pygame.draw.rect(screen, WHITE, restart_button_rect, 3)
    restart_text = font_medium.render("RESTART", True, BLACK)
    restart_text_rect = restart_text.get_rect(center=restart_button_rect.center)
    screen.blit(restart_text, restart_text_rect)
    
    menu_button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, stats_y + 220, 200, 50)
    pygame.draw.rect(screen, BLUE, menu_button_rect)
    pygame.draw.rect(screen, WHITE, menu_button_rect, 3)
    menu_text = font_small.render("MAIN MENU", True, WHITE)
    menu_text_rect = menu_text.get_rect(center=menu_button_rect.center)
    screen.blit(menu_text, menu_text_rect)
    
    return restart_button_rect, menu_button_rect

# --- STEP 15: MAIN GAME LOOP ---
running = True

while running:
    clock.tick(FPS)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if game_state == 'playing' and not game_over:
                    if ai_pause_mode:
                        ai_pause_mode = False
                        ai_advice_timer = 0
                    else:
                        game_state = 'menu'
                        init_game_variables()
                else:
                    running = False
            
            if event.key == pygame.K_p and game_state == 'playing' and not ai_pause_mode:
                paused = not paused
            
            # Request AI advice
            if event.key == pygame.K_a and game_state == 'playing' and not game_over and not paused and not ai_requesting and not ai_pause_mode:
                ai_requesting = True
                import threading
                def request_ai():
                    get_ai_advice()
                
                thread = threading.Thread(target=request_ai)
                thread.daemon = True
                thread.start()
            
            # Exit AI pause mode
            if event.key == pygame.K_SPACE and ai_pause_mode:
                ai_pause_mode = False
                ai_advice_timer = 0
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            if game_state == 'menu':
                start_button = draw_main_menu()
                if start_button.collidepoint(mouse_x, mouse_y):
                    game_state = 'playing'
                    init_game_variables()
            
            elif game_state == 'playing' and ai_pause_mode:
                # Check continue button in AI overlay
                continue_button = draw_ai_pause_overlay()
                if continue_button.collidepoint(mouse_x, mouse_y):
                    ai_pause_mode = False
                    ai_advice_timer = 0
            
            elif game_state == 'playing' and game_over:
                pass
            
            elif game_state == 'playing' and not game_over and not paused and not ai_requesting:
                if mouse_y >= GRID_ROWS * CELL_SIZE + 50 and mouse_y <= GRID_ROWS * CELL_SIZE + 90:
                    for i, tower_type in enumerate(TOWER_TYPES.keys()):
                        button_x = 10 + i * 80
                        if mouse_x >= button_x and mouse_x <= button_x + 70:
                            selected_tower_type = tower_type
                            break
                
                elif mouse_y < GRID_ROWS * CELL_SIZE:
                    row, col = get_grid_pos(mouse_x, mouse_y)
                    
                    if event.button == 1:
                        place_tower(row, col)
                    elif event.button == 3:
                        sell_tower(row, col)
    
    # Game logic
    if game_state == 'playing' and not game_over and not paused and not ai_pause_mode:
        spawn_timer += 1
        if enemies_in_wave < WAVE_SIZE and spawn_timer >= ENEMY_SPAWN_INTERVAL:
            spawn_timer = 0
            spawn_enemy()
        
        if enemies_in_wave >= WAVE_SIZE and len(enemy_list) == 0:
            wave_timer += 1
            if wave_timer >= WAVE_DELAY:
                wave_number += 1
                enemies_in_wave = 0
                wave_timer = 0
                money += 50
                stats['highest_wave'] = wave_number
        
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
        
        for tower in tower_list:
            tower.update(enemy_list)
        
        bullets_to_remove = []
        for bullet in bullet_list:
            if not bullet.active:
                bullets_to_remove.append(bullet)
                continue
            bullet.update()
        
        for bullet in bullets_to_remove:
            bullet_list.remove(bullet)
        
        particles_to_remove = []
        for particle in particle_list:
            particle.update()
            if particle.life <= 0:
                particles_to_remove.append(particle)
        
        for particle in particles_to_remove:
            particle_list.remove(particle)
        
        if base_health <= 0:
            game_over = True
            game_won = False
        
        if wave_number > 15:
            game_over = True
            game_won = True
    
    # Countdown AI advice timer
    if ai_advice_timer > 0 and not ai_pause_mode:
        ai_advice_timer -= 1
    
    # Drawing
    if game_state == 'menu':
        draw_main_menu()
    
    elif game_state == 'playing':
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
        
        if not game_over and not paused and not ai_pause_mode:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_y < GRID_ROWS * CELL_SIZE:
                draw_tower_preview(mouse_x, mouse_y)
        
        draw_ui()
        
        if ai_pause_mode:
            draw_ai_pause_overlay()
        elif paused:
            draw_pause_menu()
        
        if game_over:
            restart_button, menu_button = draw_game_over()
            mouse_pos = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:
                if restart_button.collidepoint(mouse_pos):
                    init_game_variables()
                    game_state = 'playing'
                    pygame.time.wait(200)
                elif menu_button.collidepoint(mouse_pos):
                    game_state = 'menu'
                    init_game_variables()
                    pygame.time.wait(200)
    
    pygame.display.flip()

pygame.quit()
sys.exit()