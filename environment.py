import numpy as np
import random
import pygame
import pickle

class Environment:
    def __init__(self, size=10, nZombies=8, nPresents=5, nRocks=3, load=False):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.startState = (0, 0)
        self.goalState = (size-1, size-1)
        self.nPresents = nPresents

        if load and self.load_grid():
            self.zombieStates, self.presentStates, self.rockStates = self.load_grid()
        else:
            self.zombieStates = self.place_random(nZombies)
            self.presentStates = self.place_random(nPresents, exclude=self.zombieStates)
            self.rockStates = self.place_random(nRocks, exclude=self.zombieStates + self.presentStates)
            self.save_grid()
        
        self.presentsCollected = set()
        
        for i, j in self.zombieStates:
            self.grid[i][j] = 1
        for i, j in self.presentStates:
            self.grid[i][j] = 2
        for i, j in self.rockStates:
            self.grid[i][j] = 3
    

    def place_random(self, num_items, exclude=[]):
        items = []
        while len(items) < num_items:
            i, j = random.randint(0, self.size-1), random.randint(0, self.size-1)
            
            if (i, j) not in items and (i, j) not in exclude and (i, j) != self.startState and (i, j) != self.goalState:
                items.append((i, j))
        
        return items
    

    def reset(self):
        self.currentState = self.startState
        self.presentsCollected = set()
        return self.currentState, tuple(self.presentsCollected)
    

    def step(self, action):
        status = ""
        i, j = self.currentState
        if action == 0: # CIMA
            i = max(i-1, 0)
        elif action == 1: # BAIXO
            i = min(i+1, self.size-1)
        elif action == 2: # ESQUERDA
            j = max(j-1, 0)
        elif action == 3: # DIREITA
            j = min(j+1, self.size-1)
        
        if (i, j) in self.rockStates:
            i, j = self.currentState
        
        self.currentState = (i, j)
        
        if self.currentState == self.goalState:
            if len(self.presentsCollected) == len(self.presentStates):
                reward = 10
                done = True
                status = "SAIU"
            else:
                reward = -1
                done = True
                status = "SAIU SEM TODOS PRESENTES"
        elif self.currentState in self.zombieStates:
            reward = -5
            done = True
            status = "MORTO PELO ZUMBI"
        elif self.currentState in self.presentStates and self.currentState not in self.presentsCollected:
            self.presentsCollected.add(self.currentState)
            reward = 2
            done = False
        else:
            reward = -0.1
            done = False
        
        return self.currentState, tuple(self.presentsCollected), reward, done, status


    def render(self, screen, cellSize=60):
        colors = {
            "background": (200, 200, 200),
            "robot": (0, 0, 255),
            "goal": (0, 255, 0),
            "zombie": (255, 0, 0),
            "present": (255, 215, 0),
            "wall": (100, 100, 100),
            "empty": (200, 200, 200)
        }

        screen.fill(colors["background"])

        for i in range(self.size):
            for j in range(self.size):
                color = colors["empty"]
                if (i, j) == self.currentState:
                    color = colors["robot"]
                elif (i, j) == self.goalState:
                    color = colors["goal"]
                elif self.grid[i][j] == 1:
                    color = colors["zombie"]
                elif self.grid[i][j] == 2:
                    if (i, j) not in self.presentsCollected:
                        color = colors["present"]
                elif self.grid[i][j] == 3:
                    color = colors["wall"]
                pygame.draw.rect(screen, color, (j * cellSize, i * cellSize, cellSize, cellSize))

        pygame.display.flip()

    #Miguel
    def save_grid(self):
        print("---------------------------------")
        print("SALVANDO GRID...")
        print("---------------------------------")

        gridData = {
            'zombiesStates': self.zombieStates,
            'presentsStates': self.presentStates,
            'rockStates': self.rockStates
        }

        with open('grid.pkl', 'wb') as f:
            pickle.dump(gridData, f)

    def load_grid(self):
        print("---------------------------------")
        print("CARREGANDO GRID...")
        print("---------------------------------")

        try:
            with open('grid.pkl', 'rb') as f:
                gridData = pickle.load(f)
                return gridData['zombiesStates'], gridData['presentsStates'], gridData['rockStates']
        except FileNotFoundError:
            return None