import numpy as np

class QuadraticProgram:
   def __init__(self, Q: np.ndarray, p: np.ndarray, A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray):
      self.Q = Q
      self.p = p
      self.A = A
      self.b = b
      self.C = C
      self.d = d

      self.N = Q.shape[0]
      self.M = A.shape[0]
      self.P = C.shape[0]

      assert(np.linalg.matrix_rank(A) == self.M)
      assert(np.linalg.matrix_rank(C) == self.P)

      assert(self.M < self.N)
      assert(self.P < self.N)

      assert(len(p.shape) == 1)
      assert(len(b.shape) == 1)
      assert(len(d.shape) == 1)

      assert(p.shape[0] == self.N)
      assert(b.shape[0] == self.N)
      assert(d.shape[0] == self.N)

      assert(self.N > 0)
      assert(self.N == Q.shape[1])

   def objective(self, x: np.ndarray, t: float):
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      return 0.5 * np.dot(np.dot(x, self.Q), x) + np.dot(self.p, x) - (1.0 / t) * np.sum(np.log(-(np.dot(self.C, x) - self.d)))

   def gradient(self, x: np.ndarray, t: float):
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      return np.dot(self.Q, x) - (1.0 / t) * (self.C / (np.dot(self.C, x) - self.d))

   def hessian(self, x: np.ndarray, t: float):
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      hess = np.zeros_like(self.Q)
      linear_terms = self.dot(self.C, x) - self.d
      for i in range(self.P):
         hess += np.outer(self.C[i], self.C[i]) / linear_terms[i]
      hess *= (1.0 / t)
      hess += self.Q

      return hess
