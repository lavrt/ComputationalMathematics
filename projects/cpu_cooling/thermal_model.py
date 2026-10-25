# Общий расчётный модуль: сервер и notebook используют этот класс.
import numpy as np

class ThermalModel:

    def __init__(self, L=0.05, a=0.03, P=200.0, lam=200.0, T_water=20.0):
        self.L = L
        self.a = a
        self.P = P
        self.lam = lam
        self.T_water = T_water

    def face_slice(self, face):
        s = [slice(None)] * 3
        s['xyz'.index(face[0])] = 0 if face[1] == '-' else -1
        return tuple(s)

    def cooling_weight(self, h, alpha):
        if not alpha > 0:
            raise ValueError('Нужно alpha > 0: при отсутствии теплоотвода стационарного решения нет')
        return 1.0 / (0.5 + self.lam / (alpha * h))

    def make_grid(self, N, faces, alpha):
        if N < 2 or int(N) != N or (not faces):
            raise ValueError('Нужно целое N >= 2 и хотя бы одна охлаждаемая грань')
        if len(set(faces)) != len(faces) or not set(faces) <= {'x-', 'x+', 'y-', 'y+', 'z+'}:
            raise ValueError('Допустимы разные грани x-, x+, y-, y+, z+; нижняя грань нагревается')
        h = self.L / N
        edges = np.linspace(0, self.L, N + 1)
        overlap = np.maximum(0, np.minimum(edges[1:], (self.L + self.a) / 2) - np.maximum(edges[:-1], (self.L - self.a) / 2))
        power = self.P * np.outer(overlap, overlap) / self.a ** 2
        d = np.zeros((N, N, N))
        for axis in range(3):
            s1, s2 = ([slice(None)] * 3, [slice(None)] * 3)
            s1[axis], s2[axis] = (slice(None, -1), slice(1, None))
            d[tuple(s1)] += 1
            d[tuple(s2)] += 1
        for face in faces:
            d[self.face_slice(face)] += self.cooling_weight(h, alpha)
        b = np.zeros_like(d)
        b[:, :, 0] = power / (self.lam * h)
        parity = np.indices(d.shape).sum(axis=0) % 2
        return (h, d, b, power, parity)

    def neighbors(self, u):
        s = np.zeros_like(u)
        s[1:, :, :] += u[:-1, :, :]
        s[:-1, :, :] += u[1:, :, :]
        s[:, 1:, :] += u[:, :-1, :]
        s[:, :-1, :] += u[:, 1:, :]
        s[:, :, 1:] += u[:, :, :-1]
        s[:, :, :-1] += u[:, :, 1:]
        return s

    def residual(self, u, d, b):
        return np.max(np.abs(b + self.neighbors(u) - d * u)) / np.max(np.abs(b))

    def jacobi(self, d, b, parity, eps=1e-08, maxiter=300000):
        u = np.zeros_like(b)
        history = []
        for it in range(1, maxiter + 1):
            u = (self.neighbors(u) + b) / d
            r = self.residual(u, d, b)
            history.append(r)
            if r < eps:
                return (u, np.array(history))
        raise RuntimeError('Метод Якоби не сошёлся')

    def seidel(self, d, b, parity, eps=1e-08, maxiter=300000):
        u = np.zeros_like(b)
        history = []
        masks = [parity == 0, parity == 1]
        for it in range(1, maxiter + 1):
            for mask in masks:
                candidate = (self.neighbors(u) + b) / d
                u[mask] = candidate[mask]
            r = self.residual(u, d, b)
            history.append(r)
            if r < eps:
                return (u, np.array(history))
        raise RuntimeError('Метод Зейделя не сошёлся')

    def sor(self, d, b, parity, omega=1.8, eps=1e-08, maxiter=300000):
        if not 0 < omega < 2:
            raise ValueError('Нужно 0 < omega < 2')
        u = np.zeros_like(b)
        history = []
        masks = [parity == 0, parity == 1]
        for it in range(1, maxiter + 1):
            for mask in masks:
                candidate = (self.neighbors(u) + b) / d
                u[mask] += omega * (candidate[mask] - u[mask])
            r = self.residual(u, d, b)
            history.append(r)
            if r < eps:
                return (u, np.array(history))
        raise RuntimeError('Метод релаксации не сошёлся')

    def contact_temperature(self, u, h, power):
        return self.T_water + u[:, :, 0] + power / (2 * self.lam * h)

    def wall_temperature(self, u, h, face, alpha):
        beta = self.cooling_weight(h, alpha)
        return self.T_water + (1 - beta / 2) * u[self.face_slice(face)]

    def metrics(self, u, h, power, faces, alpha):
        contact = self.contact_temperature(u, h, power)
        tmax = contact[power > 0].max()
        mean = np.sum(contact * power) / self.P
        beta = self.cooling_weight(h, alpha)
        flows = {f: float(self.lam * h * beta * u[self.face_slice(f)].sum()) for f in faces}
        return (float(tmax), float(mean), flows)
