import numpy as np
from perceptron import Perceptron

# Helpers lógicos fijando pesos a mano (sin entrenar)
class LogicPerceptron(Perceptron):
    @staticmethod
    def AND():
        p = Perceptron()
        # b = -1.5, w = [1, 1]  -> activa solo cuando x1=x2=1
        p.w_ = np.array([-1.5, 1.0, 1.0], dtype=float)
        return p

    @staticmethod
    def OR():
        p = Perceptron()
        # b = -0.5, w = [1, 1] -> activa cuando al menos uno es 1
        p.w_ = np.array([-0.5, 1.0, 1.0], dtype=float)
        return p

    @staticmethod
    def NOT():
        p = Perceptron()
        # Una sola entrada: b = 0.5, w = [-1] -> activa cuando x=0
        p.w_ = np.array([0.5, -1.0], dtype=float)
        return p

def _to_2d(X, n_features):
    """Asegura que X es (n, n_features)"""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != n_features:
        raise ValueError(f"Se esperaban {n_features} características, recibidas {X.shape[1]}")
    return X

def _bin01(y_pm1):
    """Convierte {-1,1} -> {0,1}"""
    y_pm1 = np.asarray(y_pm1)
    return np.where(y_pm1 == 1, 1, 0)

class XORGate:
    """
    XOR implementado como (OR) AND (NOT AND), usando la clase Perceptron dada.
    Entradas: X en {0,1} de shape (n,2) o (2,)
    Salida: {0,1}
    """
    def __init__(self):
        self.p_or   = LogicPerceptron.OR()
        self.p_and1 = LogicPerceptron.AND()
        self.p_not  = LogicPerceptron.NOT()
        self.p_and2 = LogicPerceptron.AND()  # AND final

    def predict(self, X):
        X = _to_2d(X, 2)              # (n, 2)
        h_or_pm1   = self.p_or.predict(X)    # {-1,1}
        h_and_pm1  = self.p_and1.predict(X)  # {-1,1}

        # Pasamos a {0,1} para conectar con NOT y AND final
        h_or  = _bin01(h_or_pm1)              # (n,)
        h_and = _bin01(h_and_pm1)             # (n,)

        # NOT es unario: darle shape (n,1)
        h_not_and_pm1 = self.p_not.predict(h_and.reshape(-1, 1))   # {-1,1}
        h_not_and = _bin01(h_not_and_pm1)                          # {0,1}

        # AND final entre (OR, NOT(AND))
        X_final = np.c_[h_or, h_not_and]       # (n,2) en {0,1}
        y_pm1 = self.p_and2.predict(X_final)   # {-1,1}
        y = _bin01(y_pm1)                      # {0,1}
        # Si entrada fue (2,), devolver escalar
        return y.item() if y.shape[0] == 1 else y

# === Demo rápido ===
if __name__ == "__main__":
    xor = XORGate()
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = xor.predict(X)
    print("X:\n", X)
    print("XOR:\n", y)  # Esperado: [0,1,1,0]