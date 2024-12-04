COMMENT
/*                                                                               
Copyright (c) 2015 EPFL-BBP, All rights reserved.                                
                                                                                 
THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''                    
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,            
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR           
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT                 
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR           
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF             
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                  
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,            
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE             
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN           
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                    
                                                                                 
This work is licensed under a                                                    
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. 
To view a copy of this license, visit                                            
http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode or send a letter to   
Creative Commons,                                                                
171 Second Street, Suite 300,                                                    
San Francisco, California, 94105, USA.                                           
*/                 
ENDCOMMENT

TITLE Probabilistic AMPA and NMDA receptor with presynaptic short-term plasticity 


COMMENT
AMPA and NMDA receptor conductance using a dual-exponential profile
presynaptic short-term plasticity as in Fuhrmann et al. 2002

_EMS (Eilif Michael Srikanth)
Modification of ProbAMPANMDA: 2-State model by Eilif Muller, Michael Reimann, Srikanth Ramaswamy, Blue Brain Project, August 2011
This new model was motivated by the following constraints:

1) No consumption on failure.  
2) No release just after release until recovery.
3) Same ensemble averaged trace as deterministic/canonical Tsodyks-Markram 
   using same parameters determined from experiment.
4) Same quantal size as present production probabilistic model.

To satisfy these constaints, the synapse is implemented as a
uni-vesicular (generalization to multi-vesicular should be
straight-forward) 2-state Markov process.  The states are
{1=recovered, 0=unrecovered}.

For a pre-synaptic spike or external spontaneous release trigger
event, the synapse will only release if it is in the recovered state,
and with probability u (which follows facilitation dynamics).  If it
releases, it will transition to the unrecovered state.  Recovery is as
a Poisson process with rate 1/Dep.

This model satisfies all of (1)-(4).
ENDCOMMENT

COMMENT                                                                          
/**                                                                              
 @file ProbAMPANMDA_EMS.mod                                                        
 @brief Probabilistic AMPA and NMDA receptor with presynaptic short-term plasticity                   
 @author Eilif Muller, Michael Reimann, Srikanth Ramaswamy, James King @ BBP     
 @date 2011                                                                      
*/                                                                               
ENDCOMMENT  

NEURON {
    THREADSAFE
        POINT_PROCESS AMPANMDA_EMS
        RANGE tau_r_AMPA, tau_d_AMPA, tau_r_NMDA, tau_d_NMDA
        RANGE i, i_AMPA, i_NMDA, g_AMPA, g_NMDA, g, e, NMDA_ratio
        RANGE A_AMPA_step, B_AMPA_step, A_NMDA_step, B_NMDA_step
        RANGE gamma
        NONSPECIFIC_CURRENT i
}

PARAMETER {


        tau_r_AMPA = 0.2   (ms)  : dual-exponential conductance profile
        tau_d_AMPA = 1.7    (ms)  : IMPORTANT: tau_r < tau_d
        tau_r_NMDA = 0.29   (ms) : dual-exponential conductance profile
        tau_d_NMDA = 43     (ms) : IMPORTANT: tau_r < tau_d

        e = 0     (mV)  : AMPA and NMDA reversal potential
        mg = 1   (mM)  : initial concentration of mg2+
        mggate
        gamma = 0.062 (/mV)
        
	   NMDA_ratio = 0.71 (1) : The ratio of NMDA to AMPA
}

COMMENT
The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1 
for comparison with Pr to decide whether to activate the synapse or not
ENDCOMMENT


ASSIGNED {

        v (mV)
        i (nA)
        i_AMPA (nA)
        i_NMDA (nA)
        g_AMPA (uS)
        g_NMDA (uS)
        g (uS)
        factor_AMPA
        factor_NMDA
        A_AMPA_step
        B_AMPA_step
        A_NMDA_step
        B_NMDA_step
        

	: Recording these three, you can observe full state of model
	: tsyn_fac gives you presynaptic times, Rstate gives you 
        : state transitions,
        : u gives you the "release probability" at transitions 
        : (attention: u is event based based, so only valid at incoming events)
	

}

STATE {

        A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA
        B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA
        A_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_r_NMDA
        B_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_d_NMDA
}

INITIAL{

        LOCAL tp_AMPA, tp_NMDA


        A_AMPA = 0
        B_AMPA = 0
        
        A_NMDA = 0
        B_NMDA = 0
        
        tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance
        tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA) :time to peak of the conductance
        
        factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak
        factor_AMPA = 1/factor_AMPA
        
        factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA) :NMDA Normalization factor - so that when t = tp_NMDA, gsyn = gpeak
        factor_NMDA = 1/factor_NMDA

        A_AMPA_step = exp(dt*(( - 1.0 ) / tau_r_AMPA))
        B_AMPA_step = exp(dt*(( - 1.0 ) / tau_d_AMPA))
        A_NMDA_step = exp(dt*(( - 1.0 ) / tau_r_NMDA))
        B_NMDA_step = exp(dt*(( - 1.0 ) / tau_d_NMDA))
}

BREAKPOINT {

        SOLVE state
        mggate = 1 / (1 + exp(gamma  * -(v)) * (mg / 3.57 (mM))) :mggate kinetics - Jahr & Stevens 1990
        g_AMPA = B_AMPA-A_AMPA :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA
        g_NMDA = (B_NMDA-A_NMDA) * mggate :compute time varying conductance as the difference of state variables B_NMDA and A_NMDA and mggate kinetics
        g = g_AMPA + g_NMDA
        i_AMPA = g_AMPA*(v-e) :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal
        i_NMDA = g_NMDA*(v-e) :compute the NMDA driving force based on the time varying conductance, membrane potential, and NMDA reversal
        i = i_AMPA + i_NMDA
}

PROCEDURE state() {
        A_AMPA = A_AMPA*A_AMPA_step
        B_AMPA = B_AMPA*B_AMPA_step
        A_NMDA = A_NMDA*A_NMDA_step
        B_NMDA = B_NMDA*B_NMDA_step
}


NET_RECEIVE (weight,weight_AMPA, weight_NMDA){
        weight_AMPA = weight
        weight_NMDA = weight * NMDA_ratio
         A_AMPA = A_AMPA + weight_AMPA*factor_AMPA
         B_AMPA = B_AMPA + weight_AMPA*factor_AMPA
         A_NMDA = A_NMDA + weight_NMDA*factor_NMDA
         B_NMDA = B_NMDA + weight_NMDA*factor_NMDA
}
