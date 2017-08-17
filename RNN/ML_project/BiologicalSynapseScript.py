# IMPORTS:
import math.exp as exp

'''
THEORETICAL DESCRIPTION:
* Look at each index number in both models to understand the sun equivalency
* We implement only the facilitation process (randomly chosen)

In the original model:
    1. h(t+1) = phi(w*h + b)
    2. x(t+1) = w*h(t+1) = w( phi(x(t)) + u )
    3. y'(j) =  - ( y(j) / theta(y)) + u*( 1 - y(j))*r(j)
    4. h(t+1) = y*h(t)

* r_j = 1 / (1 + exp((-4)*y_j))

In our model (the sub equivalent model):
    1. x' = -x + J*phi(x) + u
    2. x(t+1) = -x + J*phi(x(t)) + u
    3. y(i,t+1) = y(i,t) + alpha*( (1-y(i,t) / theta) + u*h(i,t) )
    4. h(t+1) = y*h(t)

* r_j - a constant type of function

CONSTANT SIZES:
* alpha ~ {0,1}  - usually we'll hold 0.1 0.01 as a value
* theta_y ~ 10
* u ~ 0.1
'''

### FACILITATION FUNCTION DEFINITION:
## Description:
##      This function will perform update to Y's value (namely Y(j+1))
## Parameters:
##      y_i - The current value of y
##      tetha_y - Timescale factor
##      U - Scaling factor (for the current state input)
##      r_j - The nonlinear function at step j
## Returns:
##      For our case, we denote i = j + 1
##      (1) dy_i -
##      (2) y_i -
def facilitation(y_j, h_j, x_j,):
    # denote con
    # stant factors:
    tetha_y = 10
    u = 0.1
    alpha = 0.1
    # alpha = 0.01

    # r_j:
    r_j = 1 / (1 + exp((-4)*y_j))

    # dx_i (equivalent to dx_j+1):
    dx_i = -x_j + W
    # TODO - implement function

    # x_i (equivalent to x_j+1)"
    # TODO - implement function

    # y_j+1 (equivalent to y_i):
    facilitation_element = (1 - y_j) / tetha_y
    current_state_element = u*h_j
    y_i = y_j + alpha*(facilitation_element + current_state_element)

    # h_new:
    h_new = y_i*h_j

    return [y_i, h_new]