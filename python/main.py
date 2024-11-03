import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from PIL import Image

OFF_SCREEN = 0
ON_SCREEN = 1
render_mode = OFF_SCREEN
test_case = 4

if test_case == 4:
    xml_path = '../model/swing90.xml'
else:
    xml_path = '../model/hinge_ball1.xml' #xml file (assumes this is in the same folder as this file)
sim_pause = True
next_frame = False
# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    if test_case == 0:
        data.qvel[0] = 2
        data.qvel[1] = 2
        data.qvel[2] = 0
        #initialize the controller here. This function is called once, in the beginning
        mj.mj_forward(model, data)
    #initialize the controller here. This function is called once, in the beginning
    elif test_case == 1:
        data.qvel[0] = 0;
        data.qvel[1] = 2;
        data.qvel[2] = -2;

    elif test_case == 2:
        data.qpos[0] = -np.pi / 4;
        data.qpos[1] = 0;
        data.qvel[2] = 2;

    elif test_case == 3:
        data.qpos[0] = 0;
        data.qpos[1] = np.pi / 4;
        data.qvel[2] = 2;
    elif test_case == 4:
        data.qpos[1] = np.pi / 4;
        data.qvel[0] = -2.8284;
        data.qvel[2] = 2;
    mj.mj_forward(model, data)

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
    if act == glfw.PRESS and key == glfw.KEY_SPACE:
        global sim_pause
        sim_pause = not sim_pause
    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        global next_frame
        next_frame = True

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
cam.azimuth = 90
cam.elevation = -30
cam.distance = 7
cam.lookat = np.array([0.0, 0.0, 0])

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

if render_mode == OFF_SCREEN:
    total_time = 5.7
    h = model.opt.timestep
    total_frames = int(total_time / h)

    renderer = mj.Renderer(model, width=1024, height=1024)
    num_frames = 3  # Set this to the desired number of frames

    if num_frames == 3:
        frames_to_skip = int(total_frames / 2)
    else:
        frames_to_skip = total_frames // num_frames
    for frame in range(num_frames):
        # Render the current frame to an off-screen buffer
        renderer.update_scene(data)
        img = renderer.render()

        # Convert to image format and save as PNG or JPG
        img = Image.fromarray(img)  # Flip image vertically if needed
        if num_frames == 3:
            img.save(f"frames/body_{frame:04d}.jpg", quality=95)  # Save as frame_0000.jpg, frame_0001.jpg, ...
        else:
            img.save(f"frames/frame_{frame:04d}.jpg", quality=95)

        for _ in range(frames_to_skip):
            mj.mj_step(model, data)  # Advance the simulation

    # Clean up
    print("Done!")

elif render_mode == ON_SCREEN:
    while not glfw.window_should_close(window):
        time_prev = data.time

        if sim_pause == False or next_frame == True:
            time_prev = data.time
            while (data.time - time_prev < 1.0 / 60.0):
                mj.mj_step(model, data)
            next_frame = False
            print(f"sim time: {data.time}")

        # get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mj.mjv_updateScene(model, data, opt, None, cam,
                           mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    glfw.terminate()