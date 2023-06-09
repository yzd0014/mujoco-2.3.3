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

void CharaterPDController(const mjModel* m, mjData* d)
{
    mjtNum kp = 5;
    mjtNum kd = 0.1;
    for (int i = 0; i < 21; i++)
    {
        d->ctrl[i] = kp * (0 - d->qpos[i + 7]) - kd * d->qvel[i + 6];
    }

    mjtNum rootPos[3], rootVel[3];
    mju_copy3(rootPos, &d->xipos[3]);
    mju_copy3(rootVel, &d->cvel[6 + 3]);
    
    mjtNum err[3];
    mju_sub3(err, CharaterControl::torPos, rootPos);
    //printf("%f %f %f\n", err[0], err[1], err[2]);
    mjtNum mp[3];
    mju_scl3(mp, err, kp);
    //printf("%f %f %f\n", mp[0], mp[1], mp[2]);
    mjtNum md[3];
    mju_scl3(md, rootVel, kd);
    //printf("%f %f %f\n", md[0], md[1], md[2]);
    mjtNum mf[3];
    mju_sub3(mf, mp, md);
    
    mj_jacBodyCom(m, d, CharaterControl::J, NULL, 1);
    mju_transpose(CharaterControl::J_transpose, CharaterControl::J, m->nv, 3);
    mju_mulMatVec(CharaterControl::jointTorque, CharaterControl::J_transpose, mf, m->nv, 3);
    for (int i = 0; i < 15; i++)
    {
        //d->ctrl[i] += CharaterControl::jointTorque[i + 6];
    }
}
void BallJointController(const mjModel* m, mjData* d)
{ 
    if (tickCount != -1)
    {
        std::cout << d->xpos[3] << ", " << d->xpos[4] << ", " << d->xpos[5] << std::endl;
    }
    tickCount++;
}
void InitializeController(const mjModel* m, mjData* d)
{
    mj_forward(m, d);
    /* {
        mjcb_control = CharaterPDController;
        mju_copy3(CharaterControl::torPos, &d->xipos[3]);
        CharaterControl::J = new mjtNum[3 * m->nv];
        CharaterControl::J_transpose = new mjtNum[3 * m->nv];
        CharaterControl::jointTorque = new mjtNum[m->nv];
    }*/
    {
        mjcb_control = BallJointController;
        d->qpos[0] = cos(0.5);
        d->qpos[1] = 0;
        d->qpos[2] = -sin(0.5);
        d->qpos[3] = 0;
    }
    mj_forward(m, d);
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
    m = mj_loadXML("ball_joint.xml", NULL, error, 1000);
    fs.open("../matlab/plot.csv", std::ios::out | std::ios::app);
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

    double arr_view[] = { 90, -20, 10, 0, 0.000000, 1};
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
    fs.close();
    return 0;
}