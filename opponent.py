import pygame
import sys
import numpy as np
import cv2
from datetime import datetime
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import traceback
def main():
    try:
     
        pygame.init()

       
        WIDTH, HEIGHT = 800, 600
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Learning Boss Battle - Continuous Combat")
        clock = pygame.time.Clock()

     
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 50, 50)
        GREEN = (50, 255, 50)
        BLUE = (50, 50, 255)
        YELLOW = (255, 255, 0)
        PURPLE = (150, 50, 255)

   
        if not os.path.exists('recordings'):
            os.makedirs('recordings')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(f'recordings/battle_{timestamp}.mp4', fourcc, 60.0, (WIDTH, HEIGHT))

       
        class BattleAI:
            def __init__(self):
                self.model = DecisionTreeClassifier(max_depth=5)
                self.encoder = LabelEncoder()
                self.training_data = []
                self.last_trained_frame = 0
                self.training_interval = 300  
               
            def prepare_training_data(self, log_data):
               
                df = pd.DataFrame(log_data)
               
             
                df['distance'] = np.sqrt((df['player_x']-df['boss_x'])**2 + (df['player_y']-df['boss_y'])**2)
                df['health_diff'] = df['player_health'] - df['boss_health']
               
               
                df['player_action_encoded'] = self.encoder.fit_transform(df['player_action'])
               
               
                sequence_length = 5
                for i in range(len(df)-sequence_length):
                    sequence = df.iloc[i:i+sequence_length]
                    features = {
                        'distance': sequence['distance'].mean(),
                        'health_diff': sequence['health_diff'].mean(),
                        'player_action_pattern': '-'.join(sequence['player_action']),
                        'boss_health': sequence['boss_health'].iloc[-1],
                        'player_health': sequence['player_health'].iloc[-1]
                    }
                    label = sequence['boss_action'].iloc[-1]
                    self.training_data.append((features, label))
               
                return len(self.training_data)
           
            def train_model(self):
                if len(self.training_data) < 10:
                    return False
               
         
                X = pd.DataFrame([x[0] for x in self.training_data])
                y = [x[1] for x in self.training_data]
               
             
                X['player_action_pattern'] = self.encoder.fit_transform(X['player_action_pattern'])
               
             
                self.model.fit(X, y)
                return True
           
            def predict_best_move(self, current_state):
                try:
                   
                    state_df = pd.DataFrame([current_state])
                    state_df['distance'] = np.sqrt((state_df['player_x']-state_df['boss_x'])**2 +
                                                  (state_df['player_y']-state_df['boss_y'])**2)
                    state_df['health_diff'] = state_df['player_health'] - state_df['boss_health']
                    state_df['player_action_encoded'] = self.encoder.transform([current_state['player_action']])[0]
                   
                   
                    prediction = self.model.predict(state_df[['distance', 'health_diff', 'player_action_encoded']])
                    return prediction[0]
                except:
                    return "attack"  

       
        battle_ai = BattleAI()

       
        class Player:
            def __init__(self):
                self.x = WIDTH // 2
                self.y = HEIGHT // 2
                self.radius = 20
                self.speed = 5
                self.max_health = 100
                self.health = self.max_health
                self.action = "idle"
                self.action_time = 0
                self.cooldowns = {"heal": 0, "dodge": 0, "special": 0}
                self.invincible = False
                self.action_history = []
               
            def move(self, keys):
               
                move_x = keys[pygame.K_d] - keys[pygame.K_a]
                move_y = keys[pygame.K_s] - keys[pygame.K_w]
               
                if move_x != 0 and move_y != 0:
                    move_x *= 0.7071
                    move_y *= 0.7071
                   
                self.x += move_x * self.speed
                self.y += move_y * self.speed
               
               
                self.x = max(self.radius, min(WIDTH - self.radius, self.x))
                self.y = max(self.radius, min(HEIGHT - self.radius, self.y))
               
            def update_actions(self):
                if self.action != "idle":
                    self.action_time -= 1
                    if self.action_time <= 0:
                        self.action = "idle"
                       
                for action in self.cooldowns:
                    if self.cooldowns[action] > 0:
                        self.cooldowns[action] -= 1
                       
                if self.action == "dodge" and self.action_time > 30:
                    self.invincible = True
                else:
                    self.invincible = False

        class Boss:
            def __init__(self):
                self.x = 100
                self.y = 100
                self.radius = 30
                self.speed = 3.5
                self.max_health = 200
                self.health = self.max_health
                self.action = "idle"
                self.action_time = 0
                self.attack_cooldown = 0
                self.special_cooldown = 0
                self.move_history = []
               
            def make_decision(self, player, ai_system, current_frame):
               
                if current_frame > 100 and current_frame % 10 == 0:  
                    current_state = {
                        'player_x': player.x,
                        'player_y': player.y,
                        'player_action': player.action,
                        'boss_x': self.x,
                        'boss_y': self.y,
                        'boss_health': self.health,
                        'player_health': player.health
                    }
                    predicted_action = ai_system.predict_best_move(current_state)
                   
                 
                    if self.action == "idle":
                        self.action = predicted_action
                        self.action_time = 20 if predicted_action == "attack" else 30
            def move_toward(self, player):
                dx = player.x - self.x
                dy = player.y - self.y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > 1:
                    dx /= dist
                    dy /= dist
                    self.x += dx * self.speed
                    self.y += dy * self.speed

               
            def update_actions(self, player):
                if self.attack_cooldown > 0:
                    self.attack_cooldown -= 1
                if self.special_cooldown > 0:
                    self.special_cooldown -= 1
                   
                distance = ((self.x - player.x)**2 + (self.y - player.y)**2)**0.5
               
               
                if self.action == "idle":
                    if distance < 150 and self.attack_cooldown == 0:
                        self.action = "attack"
                        self.action_time = 20
                        self.attack_cooldown = 40
                    elif distance < 200 and self.special_cooldown == 0 and player.health < 70:
                        self.action = "special"
                        self.action_time = 30
                        self.special_cooldown = 120
                else:
                    self.action_time -= 1
                    if self.action_time <= 0:
                        self.action = "idle"

       
        player = Player()
        boss = Boss()
        ai_system = BattleAI()

        running = True
        frame_count = 0
        log = []
        font = pygame.font.SysFont('Arial', 24)

       
        while running:
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
         
            keys = pygame.key.get_pressed()
         
            if keys[pygame.K_j] and player.action == "idle":
                player.action = "attack"
                player.action_time = 15
            elif keys[pygame.K_k] and player.cooldowns["dodge"] == 0 and player.action == "idle":
                player.action = "dodge"
                player.action_time = 45
                player.cooldowns["dodge"] = 90
            elif keys[pygame.K_h] and player.cooldowns["heal"] == 0:
                player.action = "heal"
                player.health = min(player.max_health, player.health + 25)
                player.cooldowns["heal"] = 180
            elif keys[pygame.K_u] and player.cooldowns["special"] == 0 and player.action == "idle":
                player.action = "special"
                player.action_time = 30
                player.cooldowns["special"] = 240
           
            player.move(keys)
            player.update_actions()
           
           
            boss.make_decision(player, ai_system, frame_count)
            boss.move_toward(player)
            boss.update_actions(player)
           
         
            distance = ((player.x - boss.x)**2 + (player.y - boss.y)**2)**0.5
            collision_distance = player.radius + boss.radius
           
            if player.action == "attack" and distance < collision_distance + 20:
                boss.health -= 1.5
               
            if boss.action == "attack" and distance < collision_distance + 15:
                if not player.invincible and player.action != "dodge":
                    player.health -= 1
                   
            if boss.action == "special" and distance < 200:
                if not player.invincible:
                    player.health -= 0.8
           
           
            if player.health <= 0 or boss.health <= 0:
                running = False
           
         
            log_entry = {
                "frame": frame_count,
                "player_x": player.x,
                "player_y": player.y,
                "player_action": player.action,
                "boss_x": boss.x,
                "boss_y": boss.y,
                "boss_action": boss.action,
                "player_health": player.health,
                "boss_health": boss.health
            }
            log.append(log_entry)
           
           
            if frame_count - ai_system.last_trained_frame > ai_system.training_interval:
                samples_added = ai_system.prepare_training_data(log)
                if samples_added > 0:
                    if ai_system.train_model():
                        print(f"AI trained on {samples_added} new samples")
                ai_system.last_trained_frame = frame_count
           
            frame_count += 1
         
            screen.fill(BLACK)
           
         
            pygame.draw.rect(screen, (50, 50, 50), (50, 50, WIDTH-100, HEIGHT-100), 2)
         
            boss_color = RED
            if boss.action == "attack":
                boss_color = (255, 150, 150)
                pygame.draw.circle(screen, (255, 100, 100), (boss.x, boss.y), boss.radius + 15, 3)
            elif boss.action == "special":
                boss_color = PURPLE
                pygame.draw.circle(screen, PURPLE, (boss.x, boss.y), 50, 2)
            pygame.draw.circle(screen, boss_color, (boss.x, boss.y), boss.radius)
           
           
            player_color = BLUE
            if player.action == "attack":
                player_color = (100, 100, 255)
                pygame.draw.line(screen, WHITE,
                               (player.x, player.y),
                               (player.x + 30 * (boss.x - player.x)/max(1, distance),
                               (player.y + 30 * (boss.y - player.y)/max(1, distance)), 3))
            elif player.action == "dodge":
                player_color = GREEN
                pygame.draw.circle(screen, (100, 255, 100), (player.x, player.y), player.radius + 10, 3)
            elif player.action == "heal":
                player_color = WHITE
                pygame.draw.circle(screen, (200, 200, 255), (player.x, player.y), player.radius + 5, 2)
            elif player.action == "special":
                player_color = YELLOW
                pygame.draw.circle(screen, YELLOW, (player.x, player.y), player.radius + 8, 3)
           
            pygame.draw.circle(screen, player_color, (player.x, player.y), player.radius)
           
           
            pygame.draw.rect(screen, (70, 70, 70), (50, 20, WIDTH-100, 20))
            pygame.draw.rect(screen, GREEN, (50, 20, (WIDTH-100) * (player.health / player.max_health), 20))
            pygame.draw.rect(screen, (70, 70, 70), (50, HEIGHT-40, WIDTH-100, 20))
            pygame.draw.rect(screen, RED, (50, HEIGHT-40, (WIDTH-100) * (boss.health / boss.max_health), 20))
           
         
            controls = font.render("WASD:Move  J:Attack  K:Dodge  H:Heal  U:Special", True, WHITE)
            screen.blit(controls, (WIDTH//2 - controls.get_width()//2, HEIGHT - 30))
           
           
            ai_status = font.render(f"AI Training Samples: {len(ai_system.training_data)}", True, WHITE)
            screen.blit(ai_status, (10, 10))
           
           
            frame = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_out.write(frame)
           
            pygame.display.flip()
            clock.tick(60)

       
        video_out.release()
     
        with open(f'recordings/battle_log_{timestamp}.csv', 'w') as f:
            f.write("frame,player_x,player_y,player_action,boss_x,boss_y,boss_action,player_health,boss_health\n")
            for entry in log:
                f.write(f"{entry['frame']},{entry['player_x']},{entry['player_y']},{entry['player_action']},")
                f.write(f"{entry['boss_x']},{entry['boss_y']},{entry['boss_action']},{entry['player_health']},{entry['boss_health']}\n")
       
       
        joblib.dump(ai_system.model, f'recordings/battle_ai_{timestamp}.joblib')
        print(f"Game data and AI model saved with timestamp: {timestamp}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()