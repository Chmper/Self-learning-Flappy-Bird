import pygame
import random
import sys
import os
import numpy as np
sys.path.append("..")
import neural_network as nn

pygame.init()
pygame.font.init()
SIZE = [400, 700]
myfont = pygame.font.SysFont('Comic Sans MS', 30)

class Bird:
    def __init__(self):
        self.brain = nn.Neural_Network(2,6) ## 2 inputs, 6 hiddens
        self.x = 50
        self.y = 350
        self.fitness = 0
        self.jump = 0
        self.jump_speed = 10
        self.gravity = 10
        self.alive = True
        self.bird_sprites = [pygame.image.load("images/0.png").convert_alpha()]

    def move(self):

        if self.y > 0:
            # handling movement while jumping
            if self.jump:
                self.sprite = 1  # change to 2.png
                self.jump_speed -= 1
                self.y -= self.jump_speed
            else:
                # regular falling (increased gravity)
                self.gravity += 0.2
                self.y += self.gravity
        else:
            # in-case where the bird reaches the top
            # of the screen
            self.jump = 0
            self.y += 3

    def make_jump(self):
        self.jump = 17
        self.gravity = 5
        self.jump_speed = 10


    def bottom_check(self):
        # bird hits the bottom = DEAD
        if self.y >= SIZE[1]:
            self.alive = False
        elif self.y <= 0:
            self.alive = False


    def get_rect(self):
        # updated bird image rectangle
        img_rect = self.bird_sprites[0].get_rect()
        img_rect[0] = self.x
        img_rect[1] = self.y
        return img_rect

class Pillar:
    def __init__(self, pos):
        # pos == True is top , pos == False is bottom
        self.pos = pos
        self.img = self.get_image()

    def get_rect(self):
        # returns the pillar image rect
        return self.img.get_rect()

    def get_image(self):
        if self.pos:  # image for the top pillar
            return pygame.image.load("images/top.png").convert_alpha()
        else:  # image for the bottom pillar
            return pygame.image.load("images/bottom.png").convert_alpha()

class Game:
    class Crew:
        def __init__(self):
            self.id = 1
            self.birds = []
            for i in range(10):
                self.birds.append(Bird())

        def is_alive(self):
            for bird in self.birds:
                if bird.alive:
                    return True
            return False

        def sort(self):
            self.birds.sort(key=lambda x: x.fitness, reverse=True)

        ### RULES ##
        ''' 1. sort the units of the current population in decreasing order by their fitness ranking
            2. select the top 4 units and mark them as the winners of the current population
            3. the 4 winners are directly passed on to the next population
            4. to fill the rest of the next population, create 6 offsprings as follows:
                - 1 offspring is made by a crossover of two best winners
                - 3 offsprings are made by a crossover of two random winners
                - 2 offsprings are direct copy of two random winners
            5. to add some variations, apply random mutations on each offspring.'''

        def next_generation(self):
            self.id += 1
            self.sort()

            self.birds[4].brain = self.birds[0].brain.crossover(self.birds[1].brain)

            for i in range(3):
                index1 = random.randint(0,3)
                index2 = random.randint(0,3)
                self.birds[5+i].brain = self.birds[index1].brain.crossover(self.birds[index2].brain)
            for i in range(2):
                self.birds[8+i].brain = self.birds[random.randint(0,3)].brain

            for i in range(6):
                self.birds[4+i].brain.mutate()

            for bird in self.birds:
                bird.fitness = 0
                bird.alive = True
                bird.y = 350


    def __init__(self):
        self.screen = pygame.display.set_mode((SIZE[0], SIZE[1]))
        self.pillar_x = 400
        self.offset = random.randint(0,200)
        self.top_p = Pillar(1)  # top pillar
        self.bot_p = Pillar(0)  # bottom pillar
        self.pillar_gap = 180 # gap between pillars, (can be randomised as well)
        self.crew = self.Crew()

    def pillar_move(self):
        # handling pillar movement in the background
        if self.pillar_x < -100:
            self.offset = random.randrange(-150, 150)
            self.passed = False
            self.pillar_x = 400
        self.pillar_x -= 5

    def get_distance(self, bird):

        if self.pillar_x > bird.x:
            return self.pillar_x - 50
        return 350

    def run(self):
        clock = pygame.time.Clock()
        done = True
        clock.tick(60)
        while done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.screen.fill((0,0,0))
            self.screen.blit(self.top_p.img, (self.pillar_x, 0 - self.pillar_gap - self.offset))
            self.screen.blit(self.bot_p.img, (self.pillar_x, 360 + self.pillar_gap - self.offset))

            text = myfont.render(str(self.crew.id), False, (255,0,0))
            self.screen.blit(text, (350, 10))

            for bird in self.crew.birds:
                if bird.alive:
                    if bird.brain.predict([bird.y - (0 - self.pillar_gap - self.offset + 500 + self.pillar_gap/2), self.get_distance(bird)])[0] > 0.5:
                        bird.make_jump()
                    self.screen.blit(bird.bird_sprites[0], (bird.x, bird.y))
                    bird.move()
                    bird.bottom_check()
                    self.collision(bird)
                    bird.fitness += 1


            self.pillar_move()

            if not self.crew.is_alive():
                self.reset()

            pygame.display.flip()

    def get_pillar_rect(self, pillar):
        # returns current pillar rectangle on display
        rect = pillar.get_image().get_rect()
        rect[0] = self.pillar_x
        if pillar.pos:
            # current rect y position for top pillar
            rect[1] = 0 - self.pillar_gap - self.offset
        else:
            # current rect y position for bottom pillar
            rect[1] = 360 + self.pillar_gap - self.offset
        return rect

    def collision(self, bird):
        top_rect = self.get_pillar_rect(self.top_p)
        bot_rect = self.get_pillar_rect(self.bot_p)
        #print(bird.get_rect())
        # collision check bird <> pillars
        if top_rect.colliderect(bird.get_rect()) or bot_rect.colliderect(bird.get_rect()):
            # print(self.bird.bird_sprites[self.bird.sprite].get_rect())
            bird.alive = False


    def reset(self):
        self.top_p = Pillar(1)
        self.bot_p = Pillar(0)
        self.pillar_x = 400
        self.offset = random.randint(0,200)
        self.crew.next_generation()



os.chdir(os.path.dirname(__file__))
if __name__ == "__main__":
    game = Game()
    game.run()
