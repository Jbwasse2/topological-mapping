import numpy as np

from rmp_nav.simulation import agent_factory, sim_renderer
from topological_nav.reachability import model_factory
from topological_nav.reachability.planning import NavGraph, NavGraphSPTM


class GetDistance:
    def __init__(self):
        self.model = self.get_model()

    def get_wp(self, model, ob, goal):
        agent = agent_factory.agents_dict[model["agent"]]()
        pu.db
        follower = model["follower"]
        return (
            follower.motion_policy.predict_waypoint(ob, goal),
       #     follower.sparsifier.predict_reachability(ob, goal),
        )

    def get_model(self):
        model = model_factory.get("model_12env_v2_future_pair_proximity_z0228")(
            device="cpu"
        )
        return model
    # Cv2 gives images in BGR, and from 0-255
    # We want RGB and from 0-1
    # Can also get list/ np array of images, this should be handled
    def cv2_to_model_im(self, im):
        im = np.asarray(im)
        assert len(im.shape) == 3 or len(im.shape) == 4
        if len(im.shape) == 3:
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            im = np.asarray(im)
            im = (im / 255).astype("float32")
        else:
            im = np.swapaxes(im, 1, 3)
            im = np.swapaxes(im, 2, 3)
            im = np.asarray(im)
            im = (im / 255).astype("float32")
        return im
    def get_distances_test(self):
        ob = np.random.rand(3, 64,64)
        goal = np.random.rand(11, 3, 64,64)
        wp = self.get_wp(self.model, ob, goal)
        return wp
if __name__ == "__main__":
    dist = GetDistance()
    print(dist.get_distances_test())
