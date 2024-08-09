import numpy as np
import math
import copy

class LQR():
    """
    """
    def init(self,Fx,Fu,Hxx,Huu,Hxu,gx,gu,Hxx_N,Huu_N,x0,N):
        ""
        # The dynamic programming notation to initialize an iLQR class
        ""
        self.Fx = Fx
        self.Fu = Fu
        self.Hxx = Hxx
        self.Huu = Huu
        self.Hxu = Hxu
        self.gx = gx
        self.gu = gu
        self.N = N
        self.n = np.shape(gx)[0] # sanity check index
        self.m = np.shape(gu)[0] # sanity check index
        self.Hxx_N = Hxx_N
        self.gx_N = Huu_N
        self.x0 = x0
    def init(self,A,B,Q,R,M,q,r,Q_N,q_N,x0,N):
        ""
        # The conventional LQR notation to initialize the iLQR class (overloading)
        ""
        self.Fx = A
        self.Fu = B
        self.Hxx = Q
        self.Huu = R
        self.Hxu = M
        self.gx = q
        self.gu = r
        self.N = N
        self.n = np.shape(gx)[0] # sanity check index
        self.m = np.shape(gu)[0] # sanity check index
        self.Hxx_N = Q_N
        self.gx_N = q_N
        self.x0 = x0

    def lqr_solve():
        # The Primal (i)LQR solve, conducts backward and forward pass to generate control, state and primal*
        P = np.zeros(np.zeros(n,n),N+1)
        p = np.zeros(np.zeros(n,1),N+1)
        P[-1] = Hxx_N
        p[-1] = gx_N
        Qx = np.zeros(np.zeros(n,1),N)
        Qu = np.zeros(np.zeros(n,1),N)
        Qxx = np.zeros(np.zeros(n,n),N)
        Qxu = np.zeros(np.zeros(n,m),N)
        Quu = np.zeros(np.zeros(m,m),N)
        K = np.zeros(np.zeros(m,n),N)
        k = np.zeros(np.zeros(m,1),N)
        for i in reversed(range(0, N)):
            Qx[i] = gx[i] + Fx.T.dot(p[i+1])
            Qu[i] = gu[i] + Fu.T.dot(p[i+1])
            Qxx[i] = Hxx[i] + np.dot(Fx.T.dot(P[i+1]),Fx)
            Qxu[i] = Hxu[i] + np.dot(Fx.T.dot(P[i+1]),Fu)
            Quu[i] = Huu[i] + np.dot(Fu.T.dot(P[i+1]),Fu)
            Quu_inv = np.linalg.inv(Quu)
            K[i] = -Quu_inv.dot(Qxu.T)
            k[i] = -Quu_inv.dot(Qu)
            P[i] = Qxx - np.dot(Qxu.dot(Quu_inv),Qxu.T)
            p[i] = Qx - np.dot(Qxu.dot(Quu_inv),Qu)
        x = np.zeros(np.zeros(n,1),N+1)
        x[0] = x0
        u = np.zeros(np.zeros(m,1),N)
        for i in range(0,N):
            u[i] = K[i].dot(x[i]) + k[i]
            x[i+1] = Fx.dot(x[i]) + Fu.dot(u[i])
        return x,u,P,p

    
