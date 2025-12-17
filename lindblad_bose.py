"""
File:           lindblad_bose.py
Author:         Spenser Talkington (spenser@upenn.edu)
Description:    library for response functions of dissipative bosonic systems
                related publication: npj Quantum Materials 9, 104 (2024)
"""

import numpy as np
from scipy.linalg import eig


class LindbladianBoseSP:
    """
    Python class for creating representation of fermionic "third-quantized" single particle lindbladians.
    Routines include
        A(),B(),C() : matrices to construct dissipative part
        coherent() : coherent part of single particle lindbladian
        dissipative() : dissipative part of single particle lindbladian (not multiplied by i Gamma)
        lindblad() : single particle lindbladian
        Xi : effective non-Hermitian Hamiltonian
        Sigma : Keldysh self energy
        GR(),GA(),GK() : retarded Green's function, GA=(GR)^dag, GK=GR.Sigma.GA
        spectral() : -1/pi * Im(Tr(G_R))
        eigs() : vals,vecsR,vecsL.conj(),vecsR.conj(),vecsL (of Xi)
        density() : orbital/site resolved density operator, includes off-diagonal coherences
        diamagnetic() : <Ja> (not multiplied by -i)
        paramagnetic() : <[Ja,Jb]> (not multiplied by -i)
        paramagnetic_integrated() : int dw <[Ja,Jb]>
        triangle() : <[Ja,[Jb,jc]]> (not multiplied by -1)
    """
    #initiate an instance of a single-particle lindbladian
    def __init__(self,n_states,ham,delta,is_nambu,*jump_ops):
        self.N = n_states
        self.H = ham
        self.Δ = delta
        self.nambu = is_nambu #the correct specification of this is crucial--is_nambu is only True if Δ(k) is non-zero or C(k) is non-zero
        self.jumps = jump_ops

        if type(is_nambu)!=bool:
            raise ValueError("is_nambu is not boolean!")

        #create larkin rotation matrix
        def larkin():
            n = self.N
            l = np.zeros((4*n,4*n),dtype=complex)
            l[0:n,0:n] = np.identity(n)
            l[n:2*n,0:n] = np.identity(n)
            l[2*n:3*n,n:2*n] = np.identity(n)
            l[3*n:4*n,n:2*n] = -np.identity(n)
            l[0:n,2*n:3*n] = np.identity(n)
            l[n:2*n,2*n:3*n] = -np.identity(n)
            l[2*n:3*n,3*n:4*n] = np.identity(n)
            l[3*n:4*n,3*n:4*n] = np.identity(n)
            return l.T/np.sqrt(2)
        self.L = larkin()

    #methods to make matrix blocks
    def A(self,k):
        n = self.N
        temp = np.zeros((n,n),dtype=complex)
        for jump in self.jumps:
            temp += np.outer(np.conjugate(jump(k)[0:n]),jump(k)[0:n])
        return temp
        
    def B(self,k):
        n = self.N
        temp = np.zeros((n,n),dtype=complex)
        for jump in self.jumps:
            temp += np.outer(jump(-k)[n:2*n],np.conjugate(jump(-k)[n:2*n]))
        return temp
        
    def C(self,k):
        n = self.N
        temp = np.zeros((n,n),dtype=complex)
        for jump in self.jumps:
            temp += np.outer(np.conjugate(jump(k)[0:n]),jump(k)[n:2*n])
        return temp

    #make coherent and dissipative parts
    def coherent(self,k):
        n = self.N
        H = self.H
        Δ = self.Δ
        l = np.zeros((4*n,4*n),dtype=complex)
        l[0:n,0:n] = H(k)
        l[n:2*n,n:2*n] = -H(k)
        l[2*n:3*n,2*n:3*n] = H(-k).T
        l[3*n:4*n,3*n:4*n] = -H(-k).T
        l[0:n,2*n:3*n] = Δ(k)
        l[n:2*n,3*n:4*n] = Δ(k)
        l[2*n:3*n,0:n] = np.conjugate(np.transpose(Δ(k)))
        l[3*n:4*n,n:2*n] = np.conjugate(np.transpose(Δ(k)))
        return l

    def dissipative(self,k):
        n = self.N
        A,B,C = self.A,self.B,self.C
        l = np.zeros((4*n,4*n),dtype=complex)
        l[0:n,0:n] = A(k)+B(k)
        l[n:2*n,0:n] = -2*B(k)
        l[2*n:3*n,0:n] = C(k)+C(-k).T
        l[3*n:4*n,0:n] = 2*C(-k).T
        l[0:n,n:2*n] = -2*A(k)
        l[n:2*n,n:2*n] = B(k)+A(k)
        l[2*n:3*n,n:2*n] = -2*C(k)
        l[3*n:4*n,n:2*n] = -C(k)-C(-k).T
        l[0:n,2*n:3*n] = np.conjugate(C(k).T+C(-k))
        l[n:2*n,2*n:3*n] = np.conjugate(-2*C(-k))
        l[2*n:3*n,2*n:3*n] = B(-k).T+A(-k).T
        l[3*n:4*n,2*n:3*n] = 2*A(-k).T
        l[0:n,3*n:4*n] = np.conjugate(2*C(k).T)
        l[n:2*n,3*n:4*n] = np.conjugate(-C(k).T-C(-k))
        l[2*n:3*n,3*n:4*n] = 2*B(-k).T
        l[3*n:4*n,3*n:4*n] = A(-k).T+B(-k).T
        return l

    #the single particle Lindbladian and its blocks
    def lindblad(self,k,G):
        l = self.L
        return l@(self.coherent(k)-1j*(G/2)*self.dissipative(k))@l.T #this is the full 4N x 4N nambu form, no sigma^3 for bosons needed--like Thompson and Kamenev, not like Rammer and Smith

    def Xi(self,k,G):
        n = self.N
        temp = self.lindblad(k,G)
        if self.nambu:
            return temp[0:2*n,2*n:4*n]
        else: #non-nambu
            return temp[0:n,2*n:3*n]
    
    def Sigma(self,k,G): #no -iG/2 out front--just block of the single particle lindbladian
        n = self.N
        temp = self.lindblad(k,G) #no factor of 2 here
        if self.nambu: #no sigma-3
            return temp[2*n:4*n,2*n:4*n]
        else: #non-nambu
            return temp[2*n:3*n,2*n:3*n]

    #green's functions
    def GR(self,k,G,w):
        n = self.N
        id = np.identity(n,dtype=complex)
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            return np.linalg.inv(w*s3-self.Xi(k,G))
        else: #non-nambu
            return np.linalg.inv(w*id-self.Xi(k,G))

    def GA(self,k,G,w):
        return np.conjugate(self.GR(k,G,w).T)

    def GK(self,k,G,w):
        return self.GR(k,G,w)@self.Sigma(k,G)@self.GA(k,G,w) #no factor of 2 here

    def spectral(self,k,G,w):
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            return -1/np.pi * np.imag(np.trace(self.GR(k,G,w) @ s3))
        else:
            return -1/np.pi * np.imag(np.trace(self.GR(k,G,w)))

    #our routine for making left and right eigenvectors
    def eigs(self,k,G):
        n = self.N
        if self.nambu:
            d = 2
            #ensure there is a steady state
            s3 = np.diag(np.ndarray.flatten(np.array([np.ones(n),-np.ones(n)])))
            vals = np.linalg.eigvals(s3@self.Xi(k,G))
            if(np.max(np.imag(vals))>0):
                print(vals)
                raise(ValueError("There is no steady state! Likely gain rate is too high."))
        else: #non-nambu
            d = 1
            #ensure there is a steady state
            vals = np.linalg.eigvals(self.Xi(k,G))
            if(np.max(np.imag(vals))>0):
                raise(ValueError("There is no steady state! Likely gain rate is too high."))
        #compute eigs (breaking degeneracy)
        break_degeneracy = 10**-14*np.ones((d*n,d*n),dtype=complex)
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            vals, vecsL, vecsR = eig(s3@self.Xi(k,G) + break_degeneracy, left=True, right=True)
        else:
            vals, vecsL, vecsR = eig(self.Xi(k,G) + break_degeneracy, left=True, right=True)
        vecsL = vecsL.conj() #****
        #adjust the phase on the eigenvectors
        onevec = np.zeros(d*n,dtype=complex)
        onevec[0] = 1
        vecsRp = np.zeros((d*n,d*n),dtype=complex)
        vecsLp = np.zeros((d*n,d*n),dtype=complex)
        for i in range(0,d*n,1):
            vecsRp[:,i] = vecsR[:,i]*np.exp(-1j*np.angle(np.dot(onevec,vecsR[:,i])))
            vecsLp[:,i] = vecsL[:,i]*np.exp(-1j*np.angle(np.dot(onevec,vecsL[:,i])))
        vecsR = vecsRp
        vecsL = vecsLp
        #normalize
        #(no need to orthogonalize since LR=D for some diagonal matrix)
        #(if it didn't we'd have to do something like an LU decomposition)
        for i in range(0,d*n,1):
            magnitude = (np.linalg.norm(vecsL[:,i]@vecsR[:,i])**2)**0.25
            vecsL[:,i] /= magnitude
            vecsR[:,i] /= magnitude
        for i in range(0,d*n,1):
            phase = np.angle(vecsL[:,i]@vecsR[:,i])
            vecsL[:,i] /= np.exp(1j*phase/2)
            vecsR[:,i] /= np.exp(1j*phase/2)
        return [vals,vecsR,vecsL.conj(),vecsR.conj(),vecsL] #ket,ket,bra,bra

    

    #response functions

    def density(self,k,G):
        E, uRket, uLket, uRbra, uLbra = self.eigs(k,G)
        denom = np.conj(E)[:,None] - E[None,:]
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            term = np.einsum("ba,am,mn,bn, ao,bp ->op",1/denom,uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRket.T,uRbra.T, optimize=True)
            return -0.5*(np.identity(2*self.N) + term) #s3@s3=s0
        else: #non-nambu
            term = np.einsum("ba,am,mn,bn, ao,bp ->op",1/denom,uLbra.T,self.Sigma(k,G),uLket.T, uRket.T,uRbra.T, optimize=True)
            return -0.5*(np.identity(self.N) + term)
        

    def diamagnetic(self,j,k,G):
        """
        <J>
        """
        E, uRket, uLket, uRbra, uLbra = self.eigs(k,G)
        denom = np.conj(E)[:,None] - E[None,:]
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            term = np.einsum("ba,am,mn,bn, bo,op,ap",1/denom,uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,s3@j(k),uRket.T, optimize=True)
            return -0.5*(np.trace(s3@j(k)) + term)
        else: #non-nambu
            term = np.einsum("ba,am,mn,bn, bo,op,ap",1/denom,uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j(k),uRket.T, optimize=True)
            return -0.5*(np.trace(j(k)) + term)

    def paramagnetic(self,j1,j2,W,k,G):
        """
        <[J1,J2]>
        """
        E, uRket, uLket, uRbra, uLbra = self.eigs(k,G)
        denom1 = np.conj(E)[:,None] - E[None,:]
        denom2 = np.conj(E)[:,None,None] - E[None,:,None] - W[None,None,:]
        denom3 = E[:,None,None] - np.conj(E)[None,:,None] + W[None,None,:]
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            term1 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,bcw->w",1/denom1,uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j1(k),uRket.T,uLbra.T,s3@j2(k),uRket.T,1/denom2,optimize=True)
            term2 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,acw->w",1/denom1,uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j2(k)@s3,uLket.T,uRbra.T,j1(k),uRket.T,1/denom3,optimize=True)
        else:
            term1 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,bcw->w",1/denom1,uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j1(k),uRket.T,uLbra.T,j2(k),uRket.T,1/denom2,optimize=True)
            term2 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,acw->w",1/denom1,uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j2(k),uLket.T,uRbra.T,j1(k),uRket.T,1/denom3,optimize=True)
        return -(term1+term2)

    def paramagnetic_integrated(self,j1,j2,k,G):
        """
        <[J1,J2]>
        the W-integral of paramagnetic
        """
        E, uRket, uLket, uRbra, uLbra = self.eigs(k,G)
        denom1 = np.conj(E)[:,None] - E[None,:]
        denom2 = np.conj(E)[:,None] - E[None,:]
        denom3 = E[:,None] - np.conj(E)[None,:]
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            term1 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,bc",1/denom1,uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j1(k),uRket.T,uLbra.T,s3@j2(k),uRket.T,1/denom2,optimize=True)
            term2 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,ac",1/denom1,uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j2(k)@s3,uLket.T,uRbra.T,j1(k),uRket.T,1/denom3,optimize=True)
        else:
            term1 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,bc",1/denom1,uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j1(k),uRket.T,uLbra.T,j2(k),uRket.T,1/denom2,optimize=True)
            term2 = np.einsum("ba,am,mn,bn, bi,ij,cj,ck,kl,al,ac",1/denom1,uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j2(k),uLket.T,uRbra.T,j1(k),uRket.T,1/denom3,optimize=True)
        return 1j*np.pi*(term1+term2) #scalars in front from contour integral

    #a->m1, b->m2, c->m3, d->m4
    def triangle(self,j1,j2,j3,W,Wp,k,G): #W and W' go together--not broadcasted to one dimension higher
        E, uRket, uLket, uRbra, uLbra = self.eigs(k,G)
        if self.nambu:
            s3 = np.kron(np.array([[1,0],[0,-1]],dtype=complex),np.identity(self.N))
            #1 term
            denom1 = E[:,None]-np.conj(E)[None,:]
            denom2 = np.conj(E)[:,None,None]-E[None,:,None]+(W+Wp)[None,None,:]
            denom3 = np.conj(E)[:,None,None]-E[None,:,None]+Wp[None,None,:]
            term1 = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, ab, caw, daw->w",uRbra.T,j1(k),uRket.T, uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j3(k)@s3,uLket.T, uRbra.T,j2(k)@s3,uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #2A term
            denom1 = E[:,None,None]-np.conj(E)[None,:,None]-W[None,None,:]
            denom2 = np.conj(E)[:,None,None] - np.conj(E)[None,:,None] + W[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + (W+Wp)[None,None,:]
            term2A = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, cbw, bdw, baw->w",uRbra.T,j1(k),uRket.T, uLbra.T,s3@j3(k)@s3,uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #2B term
            denom1 = np.conj(E)[:,None] - E[None,:]
            denom2 = np.conj(E)[:,None,None] - E[None,:,None] + Wp[None,None,:]
            denom3 = np.conj(E)[:,None,None] - np.conj(E)[None,:,None] + W[None,None,:]
            term2B = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, dc, daw, bdw->w",uRbra.T,j1(k),uRket.T, uLbra.T,s3@j3(k)@s3,uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #3A term
            denom1 = E[:,None,None] - E[None,:,None] - W[None,None,:]
            denom2 = np.conj(E)[:,None,None] - E[None,:,None] + W[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + (W+Wp)[None,None,:]
            term3A = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, caw, bcw, dcw->w",uRbra.T,j1(k),uRket.T, uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,s3@j3(k)@s3,uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #3B term
            denom1 = np.conj(E)[:,None] - E[None,:]
            denom2 = E[:,None,None] - E[None,:,None] + W[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + Wp[None,None,:]
            term3B = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, ba, acw, daw->w",uRbra.T,j1(k),uRket.T, uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,s3@j3(k)@s3,uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #4 term
            denom1 = E[:,None] - np.conj(E)[None,:]
            denom2 = np.conj(E)[:,None,None] - E[None,:,None] + Wp[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + (W+Wp)[None,None,:]
            term4 = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, cd, daw, dbw->w",uRbra.T,j1(k),uRket.T, uLbra.T,s3@j2(k),uRket.T, uLbra.T,s3@j3(k),uRket.T, uLbra.T,s3@self.Sigma(k,G)@s3,uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
        else: #non-nambu
            #1 term
            denom1 = E[:,None]-np.conj(E)[None,:]
            denom2 = np.conj(E)[:,None,None]-E[None,:,None]+(W+Wp)[None,None,:]
            denom3 = np.conj(E)[:,None,None]-E[None,:,None]+Wp[None,None,:]
            term1 = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, ab, caw, daw->w",uRbra.T,j1(k),uRket.T, uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j3(k),uLket.T, uRbra.T,j2(k),uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #2A term
            denom1 = E[:,None,None]-np.conj(E)[None,:,None]-W[None,None,:]
            denom2 = np.conj(E)[:,None,None] - np.conj(E)[None,:,None] + W[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + (W+Wp)[None,None,:]
            term2A = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, cbw, bdw, baw->w",uRbra.T,j1(k),uRket.T, uLbra.T,j3(k),uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,self.Sigma(k,G),uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #2B term
            denom1 = np.conj(E)[:,None] - E[None,:]
            denom2 = np.conj(E)[:,None,None] - E[None,:,None] + Wp[None,None,:]
            denom3 = np.conj(E)[:,None,None] - np.conj(E)[None,:,None] + W[None,None,:]
            term2B = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, dc, daw, bdw->w",uRbra.T,j1(k),uRket.T, uLbra.T,j3(k),uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,self.Sigma(k,G),uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #3A term
            denom1 = E[:,None,None] - E[None,:,None] - W[None,None,:]
            denom2 = np.conj(E)[:,None,None] - E[None,:,None] + W[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + (W+Wp)[None,None,:]
            term3A = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, caw, bcw, dcw->w",uRbra.T,j1(k),uRket.T, uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,j3(k),uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #3B term
            denom1 = np.conj(E)[:,None] - E[None,:]
            denom2 = E[:,None,None] - E[None,:,None] + W[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + Wp[None,None,:]
            term3B = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, ba, acw, daw->w",uRbra.T,j1(k),uRket.T, uLbra.T,self.Sigma(k,G),uLket.T, uRbra.T,j2(k),uRket.T, uLbra.T,j3(k),uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
            #4 term
            denom1 = E[:,None] - np.conj(E)[None,:]
            denom2 = np.conj(E)[:,None,None] - E[None,:,None] + Wp[None,None,:]
            denom3 = np.conj(E)[:,None,None] - E[None,:,None] + (W+Wp)[None,None,:]
            term4 = np.einsum("dq,qr,ar, as,st,bt, bu,uv,cv, cx,xy,dy, cd, daw, dbw->w",uRbra.T,j1(k),uRket.T, uLbra.T,j2(k),uRket.T, uLbra.T,j3(k),uRket.T, uLbra.T,self.Sigma(k,G),uLket.T, 1/denom1, 1/denom2, 1/denom3,optimize=True)
        return -(term1+term2A+term2B+term3A+term3B+term4)