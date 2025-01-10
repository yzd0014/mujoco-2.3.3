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
int testCase = -1;
double oldTime = 0;

#include <Eigen/Eigen>
using namespace Eigen;

char error[1000];
int mode = 4;

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

void CoordinateEae2Mju(mjtNum* i_v, mjtNum* o_v)
{
    o_v[0] = i_v[0];
    o_v[1] = -i_v[2];
    o_v[2] = i_v[1];
}

void CoordinateMju2Eae(mjtNum* i_v, mjtNum* o_v)
{
	o_v[0] = i_v[0];
	o_v[1] = i_v[2];
	o_v[2] = -i_v[1];
}

Quaterniond ComputeRotOffset()
{
    Matrix3d rotOffsetM;
	rotOffsetM.setIdentity();
	rotOffsetM.block<3, 1>(0, 0) = Vector3d(0, 0, -1);
	rotOffsetM.block<3, 1>(0, 1) = Vector3d(0, -1, 0);
	rotOffsetM.block<3, 1>(0, 2) = Vector3d(-1, 0, 0);
	Quaterniond rotOffset(rotOffsetM.transpose());
	return rotOffset;
}

void QuatToEuler(mjtNum i_quat[4], mjtNum o_Euler[3])
{
    Quaterniond rotOffset = ComputeRotOffset();
    Quaterniond q(i_quat[0], i_quat[1], i_quat[2], i_quat[3]);
	Quaterniond qEffective = rotOffset * q * rotOffset.inverse();
    mjtNum r11 = -2 * (qEffective.x() * qEffective.z() - q.w() * qEffective.y());
    mjtNum r12 = qEffective.w() * qEffective.w() + qEffective.x() * qEffective.x() - qEffective.y() * qEffective.y() - qEffective.z() * qEffective.z();
	mjtNum r21 = 2 * (qEffective.x() * qEffective.y() + qEffective.w() * qEffective.z());
	mjtNum r31 = -2 * (qEffective.y() * qEffective.z() - qEffective.w() * qEffective.x());
	mjtNum r32 = qEffective.w() * qEffective.w() - qEffective.x() * qEffective.x() + qEffective.y() * qEffective.y() - qEffective.z() * qEffective.z();
	o_Euler[0] = atan2(r31, r32);
	o_Euler[1] = asin(r21);
	o_Euler[2] = atan2(r11, r12);
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
    mjtNum dst[9];
    Matrix3d inertia = Matrix3d::Zero();
    mj_fullM(m, dst, d->qM);

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            inertia(i, j) = dst[i * 3 + j];
        }
    }
    mjtNum det = inertia.determinant();
    bool smallInertia = false;
    if (det < 1e-7 || d->qacc[2] > 1000000)
    {
        smallInertia = true;
        std::cout << "det: " << det << std::endl;
    }

    if (d->time <= 8 + 0.0000000001)
    {
        mjtNum h = 0.001;
        if (d->time == 0 || d->time - oldTime >= h - 0.0000000001)
        {
            /* mjtNum mPos[3] = { 0, 0, 0 };
             CoordinateMju2Eae(&d->xpos[3], mPos);
             fs << d->time << " " << mPos[0] << " " << mPos[1] << " " << mPos[2]
                 << " " << d->qpos[0] << " " << d->qpos[1] << " " << d->qpos[2] << std::endl;*/

                 /*	mjtNum euler[3];
                     QuatToEuler(&d->xquat[4], euler);
                     std::cout << "alpha " << d->qpos[0] << " m_alpha " << euler[2] << std::endl;
                     std::cout << "beta " << d->qpos[1] << " m_beta " << euler[1] << std::endl;
                     std::cout << "gamma " << d->qpos[2] << " m_gamma " << euler[0] << std::endl << std::endl;*/
            oldTime = d->time;
        }
    }
    else
    {
        start_sim = false;
    }
}

void InitializeController(const mjModel* m, mjData* d)
{
    if (testCase == 0)
    {
        d->qpos[1] = M_PI / 4;
        d->qvel[2] = 2;
    }
	else if (testCase == 1)
    {
        d->qpos[0] = -M_PI / 4;
        d->qvel[2] = 2;
    }
	else if (testCase == 2)
    {
        d->qvel[0] = 2;
        d->qvel[1] = 2;
        d->qvel[2] = 0;
    }
    else if (testCase == 3) //lock chain mimic
    {
        d->qvel[0] = 0;
        d->qvel[1] = 2;
        d->qvel[2] = -2;
    }
	else if (testCase == 4) //twist invarience for two ball joints
    {
        d->qpos[1] = M_PI / 4;
        d->qvel[0] = -2.8284;
        d->qvel[2] = 2;
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

int main(int argc, char* argv[])
{
    // load model from file and check for errors
    fs.open("../matlab/plot.csv", std::ios::out | std::ios::app);
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-example" && i + 1 < argc)
        {
			testCase = std::stoi(argv[i + 1]);
        }
    }
    if (testCase == 0)
    {
        m = mj_loadXML("example0.xml", NULL, error, 1000);
    }
    else if (testCase == 1)
    {
        m = mj_loadXML("example1.xml", NULL, error, 1000);
    }
    else if (testCase == 2)
    {
        m = mj_loadXML("example2.xml", NULL, error, 1000);
    }
    else if (testCase == 3)
    {
        m = mj_loadXML("example3.xml", NULL, error, 1000);
    }
    else if (testCase == 4)
    {
        m = mj_loadXML("example4.xml", NULL, error, 1000);
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

    double arr_view[] = { 90, -30, 7, 0, 0.000000, 0 };
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