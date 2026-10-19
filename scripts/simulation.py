"""Compatibility settings for newer MJWarp releases; preserve upstream physics."""

def configure_iteration_warnings(enabled=False):
    if enabled:
        return
    from ocbench.mjwarp.envs.manipulation import ManipulationMjWarpEnv
    original = ManipulationMjWarpEnv._make_data

    def make_data(self):
        original(self)
        # Upstream deliberately uses six iterations. Recent MJWarp prints on each
        # saturated solve, producing millions of lines. Only disable that print;
        # keep contact/capacity warnings and all solver settings unchanged.
        self._model.opt.warn_overflow &= ~int(self._mjw.OverflowType.ITERATIONS)

    ManipulationMjWarpEnv._make_data = make_data
