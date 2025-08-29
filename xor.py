import numpy as np
from perceptron import Perceptron

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
    XOR implementado como (OR) AND (NOT AND), usando perceptrones entrenados
    en tiempo de ejecución.

    Entradas: X en {0,1} de shape (n,2) o (2,)
    Salida: {0,1}
    """

    def __init__(self):
        # Perceptrones para OR, AND, NOT y el AND final
        self.p_or = Perceptron()
        self.p_and1 = Perceptron()
        self.p_not = Perceptron()
        self.p_and2 = Perceptron()

        # Entrenamiento de cada perceptrón lógico
        self._train_components()

    def _train_components(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # OR
        y_or = np.array([0, 1, 1, 1])
        self.p_or.train(X, np.where(y_or == 1, 1, -1))

        # AND sobre entradas originales
        y_and = np.array([0, 0, 0, 1])
        self.p_and1.train(X, np.where(y_and == 1, 1, -1))

        # NOT unario
        X_not = np.array([[0], [1]])
        y_not = np.array([1, 0])
        self.p_not.train(X_not, np.where(y_not == 1, 1, -1))

        # AND final para XOR: usa salidas de OR y NOT(AND)
        h_or = _bin01(self.p_or.predict(X))
        h_and = _bin01(self.p_and1.predict(X))
        h_not_and = _bin01(self.p_not.predict(h_and.reshape(-1, 1)))
        X_final = np.c_[h_or, h_not_and]
        y_xor = np.array([0, 1, 1, 0])
        self.p_and2.train(X_final, np.where(y_xor == 1, 1, -1))

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
