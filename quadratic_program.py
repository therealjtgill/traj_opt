import copy
import numpy as np
from typing import List, Tuple

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
      min x^T Q x + p^T x\n
      s.t.\n
      Ax = b\n
      Cx <= d\n
      len(x) = N\n
      A in R^(M x N)
      rank(A) = M < N\n
      C in R^(P x N)\n
      '''
      self.initialize(Q, p, A, b, C, d)

   def initialize(
      self,
      Q: np.ndarray,
      p: np.ndarray,
      A: np.ndarray,
      b: np.ndarray,
      C: np.ndarray,
      d: np.ndarray
   ):
      '''
      min x^T Q x + p^T x\n
      s.t.\n
      Ax = b\n
      Cx <= d\n
      len(x) = N\n
      A in R^(M x N)
      rank(A) = M < N\n
      C in R^(P x N)\n
      '''
      self.Q = copy.deepcopy(Q)
      self.p = copy.deepcopy(p)
      self.A = copy.deepcopy(A)
      self.b = copy.deepcopy(b)
      self.C = copy.deepcopy(C)
      self.d = copy.deepcopy(d)

      assert(len(Q.shape) == 2)
      assert(len(A.shape) == 2)
      assert(len(C.shape) == 2)
      assert(len(p.shape) == 1)
      assert(len(b.shape) == 1)
      assert(len(d.shape) == 1)

      self.N = Q.shape[0]
      self.M = A.shape[0]
      self.P = C.shape[0]

      assert(np.linalg.matrix_rank(A) == self.M)

      assert(self.M < self.N)

      assert(p.shape[0] == self.N)
      assert(b.shape[0] == self.M)
      assert(d.shape[0] == self.P)

      assert(self.N > 0)
      assert(self.N == Q.shape[1])

      # self._partial_kkt_mat = np.zeros((self.N + self.M, self.N + self.M))
      # self._partial_kkt_mat[self.N: self.N + self.M, 0: self.N] = self.A
      # self._partial_kkt_mat[0: self.N, self.N: (self.N + self.M)] = self.A.transpose()

      self.C_row_outer_products = [np.outer(row, row) for row in self.C]

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

      loss_grad = np.dot(self.Q, x) + self.p

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
         hess += self.C_row_outer_products[i] / (linear_terms[i] * linear_terms[i])
      hess *= (1.0 / t)
      hess += self.Q

      return hess

   def kkt_matrix(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the KKT matrix for this quadratic program. The KKT matrix is:\n
      \| hess(f(x) + phi(x))  A |\n
      \| A^T                  0 |\n
      where phi(x) is the sum of log barriers applied to the linear inequality
      constraints and f(x) = x^T Q x + p^T x
      '''
      kkt_mat = np.zeros((self.N + self.M, self.N + self.M))
      kkt_mat[self.N: self.N + self.M, 0: self.N] = self.A
      kkt_mat[0: self.N, self.N: (self.N + self.M)] = self.A.transpose()
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

class BoxInequalityQuadraticProgram:
   '''
   Defines a quadratic program with linear equality constraints, a positive
   semidefinite quadratic term, and inequality constraints directly on elements
   of the decision variable. Inequality constraints are added to the objective
   function using log barrier terms.
   '''
   def __init__(
      self,
      Q: np.ndarray,
      p: np.ndarray,
      A: np.ndarray,
      b: np.ndarray,
      upper_inequalities: List[Tuple[int, float]],
      lower_inequalities: List[Tuple[int, float]],
   ):
      self.initialize(Q, p, A, b, upper_inequalities, lower_inequalities)

   def initialize(
      self,
      Q: np.ndarray,
      p: np.ndarray,
      A: np.ndarray,
      b: np.ndarray,
      upper_inequalities: List[Tuple[int, float]],
      lower_inequalities: List[Tuple[int, float]],
   ):
      self.Q = copy.deepcopy(Q)
      self.p = copy.deepcopy(p)
      self.A = copy.deepcopy(A)
      self.b = copy.deepcopy(b)
      self.upper_inequalities = copy.deepcopy(upper_inequalities)
      self.lower_inequalities = copy.deepcopy(lower_inequalities)

      assert(len(Q.shape) == 2)
      assert(len(A.shape) == 2)
      assert(len(p.shape) == 1)
      assert(len(b.shape) == 1)
      assert(len(upper_inequalities) <= Q.shape[0])
      assert(len(lower_inequalities) <= Q.shape[0])

      temp_bounds = [[-np.inf, np.inf] for _ in range(Q.shape[0])]

      for index, bound in self.upper_inequalities:
         assert(index >= 0)
         assert(index <= Q.shape[0])
         temp_bounds[index][1] = bound

      for index, bound in self.lower_inequalities:
         assert(index >= 0)
         assert(index <= Q.shape[0])
         temp_bounds[index][0] = bound

      for index, lower, upper in temp_bounds:
         assert(lower < upper)

      self.N = Q.shape[0]
      self.M = A.shape[0]

      assert(np.linalg.matrix_rank(A) == self.M)

      assert(self.M < self.N)

      assert(p.shape[0] == self.N)
      assert(b.shape[0] == self.M)

      assert(self.N > 0)
      assert(self.N == Q.shape[1])

      # self._partial_kkt_mat = np.zeros((self.N + self.M, self.N + self.M))
      # self._partial_kkt_mat[self.N: self.N + self.M, 0: self.N] = self.A
      # self._partial_kkt_mat[0: self.N, self.N: (self.N + self.M)] = self.A.transpose()

   def objective(self, x: np.ndarray, t: float) -> float:
      '''
      Returns the original quadratic objective function along with the penalty
      from the barrier term applied to the linear inequality constraints.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)
      # return 0.5 * np.dot(np.dot(x, self.Q), x) + np.dot(self.p, x) - (1.0 / t) * np.sum(np.log(-(np.dot(self.C, x) - self.d)))

      beta = 0.0
      for index, upper_bound in self.upper_inequalities:
         beta += -1/t * np.log(-(x[index] - upper_bound))

      for index, lower_bound in self.lower_inequalities:
         beta += -1/t * np.log(-(lower_bound - x[index]))

      return 0.5 * np.dot(np.dot(x, self.Q), x) + np.dot(self.p, x) + beta

   def gradient(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the gradient of the original quadratic objective function along
      with the gradient of the penalty from the barrier term applied to the
      linear inequality constraints.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      # loss_grad = np.dot(self.Q, x) + self.p

      # barrier_grad = np.zeros_like(loss_grad)
      # for i in range(self.C.shape[0]):
      #    barrier_grad += self.C[i, :] / (np.dot(self.C[i, :], x) - self.d[i])
      # barrier_grad *= (-1.0 / t)

      # return loss_grad + barrier_grad

      grad = np.dot(self.Q, x) + self.p

      for index, upper_bound in self.upper_inequalities:
         grad[index] += -1/t * (1.0 / (x[index] - upper_bound))

      for index, lower_bound in self.lower_inequalities:
         grad[index] += 1/t * (1.0 / (lower_bound - x[index]))

      return grad

   def hessian(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the hessian of the original quadratic objective function along
      with the hessian of the penalty from the barrier term applied to the
      linear inequality constraints.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)
      assert(t > 0)

      # hess = np.zeros_like(self.Q)
      # linear_terms = np.dot(self.C, x) - self.d
      # for i in range(self.P):
      #    hess += self.C_row_outer_products[i] / (linear_terms[i] * linear_terms[i])
      # hess *= (1.0 / t)

      hess = copy.deepcopy(self.Q)

      for index, upper_bound in self.upper_inequalities:
         hess[index, index] += 1/t * (1.0 / (x[index] - upper_bound) ** 2)

      for index, lower_bound in self.lower_inequalities:
         hess[index, index] += 1/t * (1.0 / (lower_bound - x[index]) ** 2)

      return hess

   def kkt_matrix(self, x: np.ndarray, t: float) -> np.ndarray:
      '''
      Returns the KKT matrix for this quadratic program. The KKT matrix is:\n
      \| hess(f(x) + phi(x))  A |\n
      \| A^T                  0 |\n
      where phi(x) is the sum of log barriers applied to the linear inequality
      constraints and f(x) = x^T Q x + p^T x
      '''
      kkt_mat = np.zeros((self.N + self.M, self.N + self.M))
      kkt_mat[self.N: self.N + self.M, 0: self.N] = self.A
      kkt_mat[0: self.N, self.N: (self.N + self.M)] = self.A.transpose()
      kkt_mat[0: self.N, 0: self.N] = self.hessian(x, t)
      return kkt_mat

   def in_domain(self, x: np.ndarray) -> bool:
      '''
      Returns `True` if Cx <= d, where C and d define the linear inequality
      constraints in the quadratic program definition.
      '''
      assert(len(x.shape) == 1)
      assert(x.shape[0] == self.N)

      for index, upper_bound in self.upper_inequalities:
         if x[index] > upper_bound:
            return False

      for index, lower_bound in self.lower_inequalities:
         if x[index] < lower_bound:
            return False

      return True
