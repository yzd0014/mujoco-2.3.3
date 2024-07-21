#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <fstream>
#include <functional>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <cmath>

#include <mujoco/mujoco.h>
#include <mujoco/mjxmacro.h>
#include "simulate.h"
#include "glfw_adapter.h"
#include "array_safety.h"

std::fstream fs;
int testCase = 2;

//#include <Eigen/Eigen>
//using namespace Eigen;

char error[1000];
int mode = 0;

mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

bool start_sim = false;
bool next_step = false;
unsigned int key_s_counter = 0;

namespace CharaterControl
{
    mjtNum torPos[3];
    mjtNum* J;
    mjtNum* J_transpose;
    mjtNum* jointTorque;
}
int tickCount = -1;

void CoordinateTranslation(mjtNum* i_v, mjtNum* o_v)
{
    o_v[0] = i_v[0];
    o_v[1] = -i_v[2];
    o_v[2] = i_v[1];
}

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
    {
        mj_resetData(m, d);
        mj_forward(m, d);
        start_sim = false;
    }
    if (act == GLFW_PRESS && key == GLFW_KEY_SPACE)
    {
        start_sim = !start_sim;
    }
    if (key == GLFW_KEY_S)
    {
        key_s_counter++;
        //std::cout << key_s_counter << std::endl;
        if (act == GLFW_PRESS)
            next_step = true;
        if (key_s_counter >= 4)
        {
            key_s_counter = 4;
            next_step = true;
        }
        if (act == GLFW_RELEASE)
        {
            key_s_counter = 0;
        }
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void Tick(const mjModel* m, mjData* d)
{
    if (testCase == 0 || testCase == 1)
    {
        mjtNum vel[3];
        mju_quat2Vel(vel, d->qpos, 1);
        mjtNum vel_norm = mju_norm3(vel);
        //std::cout << "rot_norm: " << vel_norm << std::endl;

        //std::cout << m->body_inertia[3] << " " << m->body_inertia[4] << " " << m->body_inertia[5] << std::endl;
        mjtNum i_quat[3] = { d->qpos[1], d->qpos[2],  d->qpos[3] };
        mjtNum o_quat[3];
        CoordinateTranslation(i_quat, o_quat);
        //std::cout << d->qpos[0] << " " << o_quat[0] << " " << o_quat[1] << " " << o_quat[2] << std::endl;

        //std::cout << d->qvel[0] << " " << d->qvel[1] << " " << d->qvel[2] << std::endl;
    }
}

void InitializeController(const mjModel* m, mjData* d)
{
    if (testCase == 0)
    {
        d->qvel[0] = -2;
        d->qvel[1] = 0;
        d->qvel[2] = 2;
    }
    else if (testCase == 1)
    {
        mjtNum rotQuat[4];
        mjtNum targetVec[3] = { 0, 1, 1 };
        //mjtNum targetVec[3] = { 0, 0, 1 };
        mju_quatZ2Vec(rotQuat, targetVec);
        mjtNum rotMat[9];
        mju_quat2Mat(rotMat, rotQuat);
   
        mjtNum localW[3] = { 0, 0, 1 };
        mjtNum globalW[3];
        mju_rotVecMat(globalW, localW, rotMat);
        //std::cout << globalW[0] << " " << globalW[1] << " " << globalW[2] << std::endl;

        for (int i = 0; i < 4; i++) d->qpos[i] = rotQuat[i];
        for (int i = 0; i < 3; i++) d->qvel[i] = localW[i];
    }
    
    mj_forward(m, d);
    mjcb_control= Tick;
}

void CleanController()
{
    delete CharaterControl::J;
    delete CharaterControl::J_transpose;
    delete CharaterControl::jointTorque;
}

int main(void)
{
    // load model from file and check for errors
    //m = mj_loadXML("humanoid.xml", NULL, error, 1000);
    if (testCase == 0 || testCase == 1)
    {
        m = mj_loadXML("single_ball_joint.xml", NULL, error, 1000);
    }
    else if (testCase == 2)
    {
        m = mj_loadXML("lock_chain.xml", NULL, error, 1000);
    }

    if (!m)
    {
        printf("%s\n", error);
        return 1;
    }

    // make data corresponding to model
    d = mj_makeData(m);

    // init GLFW, create window, make OpenGL context current, request v-sync
    // init GLFW
    if (!glfwInit())
        mju_error("Could not initialize GLFW");
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    //mjv_defaultPerturb(&pert);
    mjv_defaultScene(&scn);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 1000);
    mjr_makeContext(m, &con, mjFONTSCALE_100);

    // ... install GLFW keyboard and mouse callbacks
     // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    double arr_view[] = { 90, -20, 10, 0, 0.000000, 1 };
    cam.azimuth = arr_view[0];
    cam.elevation = arr_view[1];
    cam.distance = arr_view[2];
    cam.lookat[0] = arr_view[3];
    cam.lookat[1] = arr_view[4];
    cam.lookat[2] = arr_view[5];

    InitializeController(m, d);
    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
        if (start_sim || next_step)
        {
            // advance interactive simulation for 1/60 sec
            //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
            //  this loop will finish on time for the next frame to be rendered at 60 fps.
            //  Otherwise add a cpu timer and exit this loop when it is time to render.
            mjtNum simstart = d->time;
            while (d->time - simstart < 1.0 / 60.0)
            {
                mj_step(m, d);
            }
            next_step = false;
        }
        if (tickCount >= 5000) break;
        // get framebuffer viewport
        mjrRect viewport = { 0, 0, 0, 0 };
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }
    CleanController();

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    mj_deleteData(d);

    return 0;
}