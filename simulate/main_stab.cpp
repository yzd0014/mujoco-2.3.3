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
#include <helper.h>
#include <Eigen/Eigen>

using namespace Eigen;

char error[1000];
double oldTime = 0;

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

int tickCount = -1;

mjtNum totalEnergy0 = 0;
mjtNum* dst = nullptr;
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
	std::cout << "total energy " << d->energy[0] + d->energy[1] << std::endl << std::endl;
}

void StabilizationController(const mjModel* m, mjData* d)
{
    //d->qvel[0] = d->qvel[0] + 0.01;
    VectorXd qdot;
    qdot.resize(m->nv);
    CopyVectorMjtoEigen(d->qvel, qdot, m->nv);
	//std::cout << "qdot before: " << qdot.transpose() << std::endl;

    int energeMomentumConstraintDim = 1;
    int nq = m->nv;
    int n = nq + energeMomentumConstraintDim;

    VectorXd mq(nq);
    mq.setZero();
    mq.segment(0, m->nv) = qdot;

    MatrixXd grad_C(energeMomentumConstraintDim, nq);
    grad_C.setZero();

	mj_fullM(m, dst, d->qM);
    MatrixXd Mr(m->nv, m->nv);
    CopyMatrixMjtoEigen(dst, Mr, m->nv, m->nv);

    MatrixXd DInv;
    DInv = Mr.inverse();

    mjtNum kineticEnergyExpected = totalEnergy0 - d->energy[0];

    mjtNum energyErr = 1.0;
    MatrixXd C(energeMomentumConstraintDim, 1);
    MatrixXd lambdaNew(energeMomentumConstraintDim, 1);
    int iter = 0;
    while (energyErr > 1e-3)
    {
        C(0, 0) = 0.5 * (mq.segment(0, m->nv).transpose() * Mr * mq.segment(0, m->nv))(0, 0) - kineticEnergyExpected;
		//std::cout << "C: " << C << std::endl;
        grad_C.block(0, 0, 1, m->nv) = (Mr * mq.segment(0, m->nv)).transpose();
		//std::cout << "grad_C: " << grad_C << std::endl;

		MatrixXd K = grad_C * DInv * grad_C.transpose();
        if (K.determinant() < 1e-10)
        {
            std::cout << "singularity in energy constraint" << std::endl;
            break;
		}
        lambdaNew = K.inverse() * C;
        //std::cout << "lambdaNew: " << lambdaNew << std::endl;
        mq = mq - DInv * grad_C.transpose() * lambdaNew;
		//std::cout << "mq: " << mq.transpose() << std::endl;

        //ForwardAngularAndTranslationalVelocity(mq);
		CopyVectorEigentoMj(mq, d->qvel, m->nv);
        mj_energyVel(m, d);
        energyErr = fabs(d->energy[1] - kineticEnergyExpected);
        iter++;
    }
    qdot = mq;
	CopyVectorEigentoMj(qdot, d->qvel, m->nv);
    std::cout << "energy constraint iter: " << iter <<  "error: " << energyErr << std::endl;
}

void InitializeController(const mjModel* m, mjData* d)
{
    dst = new mjtNum[m->nv * m->nv];
  
    mj_resetDataKeyframe(m, d, 0);
	mj_forward(m, d);
	totalEnergy0 = d->energy[0] + d->energy[1];

    mjcb_control = Tick;
    mjcb_fepr = StabilizationController;
}

void CleanController()
{
    delete[] dst;
}

int main(int argc, char* argv[])
{
    // load model from file and check for errors
    m = mj_loadXML("stab/ball_joints_5.xml", NULL, error, 1000);

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
        //if (tickCount >= 5000) break;
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

    // close GLFW, free visualization storage
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    CleanController();
    mj_deleteData(d);
    
    return 0;
}