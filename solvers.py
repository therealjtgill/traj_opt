import numpy as np
from quadratic_program import QuadraticProgram

class InfeasibleStartNewton:
   def __init__(self, qp: QuadraticProgram):
      self.qp = qp

   def residual(self, x: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
      return np.hstack(
         [
            self.qp.gradient(x, t) + np.dot(self.qp.A.transpose(), self.v),
            np.dot(self.qp.A, x) - self.qp.b
         ]
      )

   def solve(self, x_init: np.ndarray, v_init: np.ndarray, max_num_iters: int):
      assert(len(v_init.shape) == 1)
      assert(v_init.shape[0] == self.qp.M)
