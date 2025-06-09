import numpy as np

class QuadraticProgram:
   '''
   Defines a quadratic program with linear equality constraints, a positive
   semidefinite quadratic term, and linear inequality constraints. Inequality
   constraints are added to the objective function using log barrier terms.
   '''
   def __init__(
      self,
      Q: np.ndarray,
      p: np.ndarray,
      A: np.ndarray,
      b: np.ndarray,
      C: np.ndarray,
      d: np.ndarray
   ):
      '''
      min x^T Q x + p^Tx\n
      s.t. Ax = b\n
           Cx <= d\n
           len(x) = N\n
           rank(A) = M < N\n
           C in [P, N], P < N\n
      '''
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

      assert(self.M < self.N)
      assert(self.P < self.N)

      assert(len(p.shape) == 1)
      assert(len(b.shape) == 1)
      assert(len(d.shape) == 1)

      assert(p.shape[0] == self.N)
      assert(b.shape[0] == self.M)
      assert(d.shape[0] == self.P)

      assert(self.N > 0)
      assert(self.N == Q.shape[1])

      print("C shape: ", C.shape)

      self._partial_kkt_mat = np.zeros((self.N + self.M, self.N + self.M))
      self._partial_kkt_mat[self.N: self.N + self.M, 0: self.N] = self.A
      self._partial_kkt_mat[0: self.N, self.N: (self.N + self.M)] = self.A.transpose()

   def objective(self, x: np.ndarray, t: float) -> float:
      '''
      Returns the original quadratic objective function along with the penalty
      from the barrier term applied to the linear inequality constraints.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      return 0.5 * np.dot(np.dot(x, self.Q), x) + np.dot(self.p, x) - (1.0 / t) * np.sum(np.log(-(np.dot(self.C, x) - self.d)))

   def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the gradient of the original quadratic objective function along
      with the gradient of the penalty from the barrier term applied to the
      linear inequality constraints.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      loss_grad = np.dot(self.Q, x)

      barrier_grad = np.zeros_like(loss_grad)
      for i in range(self.C.shape[0]):
         barrier_grad += self.C[i, :] / (np.dot(self.C[i, :], x) - self.d[i])
      barrier_grad *= (-1.0 / t)

      return loss_grad + barrier_grad

   def hessian(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the hessian of the original quadratic objective function along
      with the hessian of the penalty from the barrier term applied to the
      linear inequality constraints.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      hess = np.zeros_like(self.Q)
      linear_terms = np.dot(self.C, x) - self.d
      for i in range(self.P):
         hess += np.outer(self.C[i], self.C[i]) / linear_terms[i]
      hess *= (1.0 / t)
      hess += self.Q

      return hess

   def kkt_matrix(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the KKT matrix for this quadratic program. The KKT matrix is:\n
      | hess(f(x) + phi(x))  A |\n
      | A^T                  0 |\n
      where phi(x) is the sum of log barriers applied to the linear inequality
      constraints and f(x) = x^T Q x + p^T x
      '''
      kkt_mat = np.copy(self._partial_kkt_mat)
      kkt_mat[0: self.N, 0: self.N] = self.hessian(x, t)
      return kkt_mat

   def in_domain(self, x: np.ndarray) -> bool:
      '''
      Returns `True` if Cx <= d, where C and d define the linear inequality
      constraints in the quadratic program definition.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)

      return np.all(np.dot(self.C, x) <= self.d)
