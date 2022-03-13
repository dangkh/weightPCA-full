import pygame
import numpy as np
import time
import transforms3d.euler as euler
from amc_parser import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class Viewer:
  def __init__(self, joints=None, motions=None):
    """
    Display motion sequence in 3D.

    Parameter
    ---------
    joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.joints = joints
    self.motions = motions
    self.frame = 0 # current frame of the motion sequence
    self.playing = False # whether is playing the motion sequence
    self.fps = 120 # frame rate

    # whether is dragging
    self.rotate_dragging = False
    self.translate_dragging = False
    # old mouse cursor position
    self.old_x = 0
    self.old_y = 0
    # global rotation
    self.global_rx = 0
    self.global_ry = 0
    # rotation matrix for camera moving
    self.rotation_R = np.eye(3)
    # rotation speed
    self.speed_rx = np.pi / 90
    self.speed_ry = np.pi / 90
    # translation speed
    self.speed_trans = 0.25
    self.speed_zoom = 0.5
    # whether the main loop should break
    self.done = False
    # default translate set manually to make sure the skeleton is in the middle
    # of the window
    # if you can't see anything in the screen, this is the first parameter you
    # need to adjust
    self.default_translate = np.array([0, -20, -100], dtype=np.float32)
    self.translate = np.copy(self.default_translate)

    pygame.init()
    self.screen_size = (1024, 768)
    self.screen = pygame.display.set_mode(
      self.screen_size, pygame.DOUBLEBUF | pygame.OPENGL
    )
    pygame.display.set_caption(
      'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
    )
    self.clock = pygame.time.Clock()

    glClearColor(1, 1, 1, 0)
    glShadeModel(GL_SMOOTH)
    glMaterialfv(
      GL_FRONT, GL_SPECULAR, np.array([1, 1, 1, 1], dtype=np.float32)
    )
    glMaterialfv(
      GL_FRONT, GL_SHININESS, np.array([100.0], dtype=np.float32)
    )
    glMaterialfv(
      GL_FRONT, GL_AMBIENT, np.array([0.7, 0.7, 0.7, 0.7], dtype=np.float32)
    )
    glEnable(GL_POINT_SMOOTH)

    glLightfv(GL_LIGHT0, GL_POSITION, np.array([1, 1, 1, 0], dtype=np.float32))
    glEnable(GL_LIGHT0)
    # glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, (self.screen_size[0]/self.screen_size[1]), 0.1, 500.0)

    glPointSize(10)
    glLineWidth(2.5)

  def process_event(self):
    """
    Handle user interface events: keydown, close, dragging.

    """
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RETURN: # reset camera
          self.translate = self.default_translate
          self.global_rx = 0
          self.global_ry = 0
        elif event.key == pygame.K_SPACE:
          self.playing = not self.playing
      elif event.type == pygame.MOUSEBUTTONDOWN: # dragging
        if event.button == 1:
          self.rotate_dragging = True
        else:
          self.translate_dragging = True
        self.old_x, self.old_y = event.pos
      elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
          self.rotate_dragging = False
        else:
          self.translate_dragging = False
      elif event.type == pygame.MOUSEMOTION:
        if self.translate_dragging:
          # haven't figure out best way to implement this
          pass
        elif self.rotate_dragging:
          new_x, new_y = event.pos
          self.global_ry -= (new_x - self.old_x) / \
              self.screen_size[0] * np.pi
          self.global_rx -= (new_y - self.old_y) / \
              self.screen_size[1] * np.pi
          self.old_x, self.old_y = new_x, new_y
    pressed = pygame.key.get_pressed()
    # rotation
    if pressed[pygame.K_DOWN]:
      self.global_rx -= self.speed_rx
    if pressed[pygame.K_UP]:
      self. global_rx += self.speed_rx
    if pressed[pygame.K_LEFT]:
      self.global_ry += self.speed_ry
    if pressed[pygame.K_RIGHT]:
      self.global_ry -= self.speed_ry
    # moving
    if pressed[pygame.K_a]:
      self.translate[0] -= self.speed_trans
    if pressed[pygame.K_d]:
      self.translate[0] += self.speed_trans
    if pressed[pygame.K_w]:
      self.translate[1] += self.speed_trans
    if pressed[pygame.K_s]:
      self.translate[1] -= self.speed_trans
    if pressed[pygame.K_q]:
      self.translate[2] += self.speed_zoom
    if pressed[pygame.K_e]:
      self.translate[2] -= self.speed_zoom
    # forward and rewind
    if pressed[pygame.K_COMMA]:
      self.frame -= 1
      if self.frame < 0:
        self.frame = len(self.motions) - 1
    if pressed[pygame.K_PERIOD]:
      self.frame += 1
      if self.frame >= len(self.motions):
        self.frame = 0
    # global rotation
    grx = euler.euler2mat(self.global_rx, 0, 0)
    gry = euler.euler2mat(0, self.global_ry, 0)
    self.rotation_R = grx.dot(gry)

  def set_joints(self, joints):
    """
    Set joints for viewer.

    Parameter
    ---------
    joints: Dict returned from `amc_parser.parse_asf`. Keys are joint names and
    values are instance of Joint class.

    """
    self.joints = joints

  def set_motion(self, motions):
    """
    Set motion sequence for viewer.

    Paramter
    --------
    motions: List returned from `amc_parser.parse_amc. Each element is a dict
    with joint names as keys and relative rotation degree as values.

    """
    self.motions = motions


  def draw_chess(self, x1, x2, x3, x4):
    # print(x1, x2, x3, x4)
    glBegin(GL_QUADS)
    glColor3f(0, 0, 0)
    tmp1 = np.array(x1).dot(self.rotation_R) + self.translate
    glVertex3f(*tmp1)
    tmp2 = np.array(x2).dot(self.rotation_R) + self.translate
    glVertex3f(*tmp2)
    tmp3 = np.array(x3).dot(self.rotation_R) + self.translate
    glVertex3f(*tmp3)
    tmp4 = np.array(x4).dot(self.rotation_R) + self.translate
    glVertex3f(*tmp4)
    glEnd()
    glFlush()

  def draw(self):
    """
    Draw the skeleton with balls and sticks.

    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ground_vertices = (
    [-100,0,-100],
    [100,0,-100],
    [100,0,100],
    [-100,0,100],

    )
# draw chessboard
    for i in range(10):
      for j in range(10):
        ii = 10*(i*2)
        jj = 10*(j*2)
        x = -100
        z = -100
        y = 0
        self.draw_chess([x+ii,y,z+jj], [x+ii+10,y,z+jj], [x+ii+10,y,z+jj+10], [x+ii,y,z+jj+10])

    glBegin(GL_POINTS)
    for j in joints.values():
      coord = np.array(
        np.squeeze(j.coordinate).dot(self.rotation_R) + \
        self.translate, dtype=np.float32
      )
      glColor3f(0,0,0)
      glVertex3f(*coord)
      print(coord)
    glEnd()
    halt
    glBegin(GL_LINES)
    for j in joints.values():
      child = j
      parent = j.parent
      if parent is not None:
        coord_x = np.array(
          np.squeeze(child.coordinate).dot(self.rotation_R)+self.translate,
          dtype=np.float32
        )
        coord_y = np.array(
          np.squeeze(parent.coordinate).dot(self.rotation_R)+self.translate,
          dtype=np.float32
        )
        glColor3f(0,0,0)
        glVertex3f(*coord_x)
        glVertex3f(*coord_y)
    glEnd()
    glutSwapBuffers()


  def draw_2(self, draw_matrix):
    """
    Draw the skeleton with balls and sticks.

    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ground_vertices = (
    [-100,0,-100],
    [100,0,-100],
    [100,0,100],
    [-100,0,100],

    )
    glBegin(GL_QUADS)
    for vertex in ground_vertices:
        tmp = np.array(vertex).dot(self.rotation_R) + self.translate
        glColor3f(0, 0.5, 0.5)
        glVertex3f(*tmp)
    glEnd()
    glFlush()

    # glBegin(GL_POINTS)
    # for j in joints.values():
    #   coord = np.array(
    #     np.squeeze(j.coordinate).dot(self.rotation_R) + \
    #     self.translate, dtype=np.float32
    #   )
    #   glColor3f(0,0,0)
    #   glVertex3f(*coord)
    #   # print(coord)
    # glEnd()
    print(self.frame)
    print(draw_matrix)
    counter = 0
    glBegin(GL_POINTS)
    while counter < len(draw_matrix):
      print(counter)
      x = draw_matrix[counter]
      y = draw_matrix[counter+1]
      z = draw_matrix[counter+2]
      counter += 3
      print(x,y,z)
      print(np.squeeze([x,y,z]).dot(self.rotation_R))
      coord = np.array(
        np.squeeze([x,y,z]).dot(self.rotation_R) + self.translate, dtype=np.float32
      )

      glColor3f(0,0,0)
      glVertex3f(*coord)
      print(coord)
    glEnd()
    glutSwapBuffers()




  def run(self):
    """
    Main loop.

    """
    while not self.done:
      self.process_event()
      self.joints['root'].set_motion(self.motions[self.frame])
      if self.playing:
        self.frame += 1
        if self.frame >= len(self.motions):
          self.frame = 0
      self.draw()
      pygame.display.set_caption(
        'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
      )
      pygame.display.flip()
      self.clock.tick(self.fps)
    pygame.quit()


  def run_2(self, options, ResultMatrix):
    """
    Main loop.

    """
    while not self.done:
      self.process_event()
      # self.joints['root'].set_motion(self.motions[self.frame])
      if self.playing:
        self.frame += 1
        if self.frame >= len(ResultMatrix):
          self.frame = 0
      self.draw_2(ResultMatrix[self.frame])
      pygame.display.set_caption(
        'AMC Parser - frame %d / %d' % (self.frame, len(self.motions))
      )
      pygame.display.flip()
      self.clock.tick(self.fps)
    pygame.quit()




if __name__ == '__main__':
  asf_path = './data/01.asf'
  amc_path = './data/01_01.amc'
  joints = parse_asf(asf_path)
  motions = parse_amc(amc_path)
  # save file 
  # f = open("myfile.txt", "w")
  # for index in range(len(motions)):
  #   joints['root'].set_motion(motions[index])
  #   for j in joints.values():
  #     coord = np.array(
  #       np.squeeze(j.coordinate).dot(np.eye(3)) + \
  #       np.array([0, -20, -100], dtype=np.float32), dtype=np.float32
  #     )
  #     f.write(str(coord[0])+ ", "+ str(coord[1])+ ", "+ str(coord[2])+", ")
  #   f.write("\n")
  # f.close()
  v = Viewer(joints, motions)
  v.run()



