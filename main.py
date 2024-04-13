import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax
import imageio


n = 10000       # number of bodies
g = .0001       # gravitational constant
m = 1.          # mass of the bodies
radius = .1     # radius of the bodies, forces will naturally soften when two bodies 'overlap'

num_timesteps = 400
timestep = .01
ts = jnp.arange(num_timesteps) * timestep
key = jrandom.PRNGKey(7)

# circle
# keys = jrandom.split(key, 2)
# theta = jrandom.uniform(keys[0], (n,)) * 2 * jnp.pi
# r = jnp.sqrt(jrandom.uniform(keys[1], (n,))) * .8
# x_init = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)], axis=-1)
# v_init = jnp.stack([- r * jnp.sin(theta), r * jnp.cos(theta)], axis=-1) * .8

# square
# x_init = jrandom.uniform(key, (n, 2), minval=-.8, maxval=.8)
# v_init = jnp.zeros((n, 2))

# chess board
x_init = jrandom.uniform(key, (2 * n, 2), minval=-.8, maxval=.8)
x_init = x_init[(jnp.floor(x_init[:, 0] / .2) + jnp.floor(x_init[:, 1] / .2)) % 2 == 1]
n = len(x_init)
v_init = jnp.zeros((n, 2))


def potential_energy(x):
    i, j = jnp.tril_indices(n, -1)
    r = jnp.sqrt(((x[i] - x[j]) ** 2).sum(-1))
    energies = jnp.where(
        r > radius,
        - g * m * m / r,
        g * m * m / (2 * radius ** 3) * (r ** 2 - 3 * radius ** 2))
    return energies.sum()


def f(t, state, args):
    x, v = state
    force = jax.grad(potential_energy)(x)
    a = - force / m
    return (v, a)


def smooth(x):
    z = jnp.linspace(-2., 2., 3)
    kernel = jnp.exp(- (z[None, :] ** 2 + z[:, None] ** 2))
    return jax.scipy.signal.convolve(x, kernel, mode='same')


def colormap(x):
    background = jnp.array([253, 252, 220])
    point = jnp.array([18, 18, 18])

    x = jnp.clip(x * .5, 0, 1)
    x = (1 - x[..., None]) * background + x[..., None] * point
    return jnp.clip(x, 0, 255).astype(jnp.uint8)


@jax.jit
def render(x):
    canvas, _, _ = jnp.histogram2d(x[:, 0], x[:, 1], bins=(1024, 1024), range=[[-1., 1.], [-1., 1.]])
    canvas = smooth(canvas.astype(jnp.float32))
    canvas = colormap(canvas)
    return canvas


solution = diffrax.diffeqsolve(diffrax.ODETerm(f), diffrax.Dopri5(), t0=ts[0], t1=ts[-1], dt0=ts[1]-ts[0], y0=(x_init, v_init), saveat=diffrax.SaveAt(ts=ts), max_steps=None)
xs, vs = solution.ys

writer = imageio.get_writer('n-body.mp4', fps=24)
for x in xs:
    frame = render(x)
    writer.append_data(np.asarray(frame))
writer.close()
