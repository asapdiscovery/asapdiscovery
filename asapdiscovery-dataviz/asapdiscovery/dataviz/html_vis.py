from .html_blocks import visualisation_header, colour_sars2, colour_mers, colour_7ene, orient_tail_sars2, orient_tail_mers, orient_tail_7ene


class HTMLVisualiser:
    allowed_targets = ("sars2", "mers", "7ene")

    def __init__(self, poses, paths, target):
        self.poses = poses
        if target not in self.allowed_targets:
            raise ValueError("Target must be one of: {}".format(self.allowed_targets))
        self.target = target


    @staticmethod
    def write_html(html, path):
        with open(path, "w") as f:
            f.write(html)

    def write_pose_visualisations(self):
        for pose, path in zip(self.poses, self.paths):
            self.write_pose_visualisation(pose, path)

    
    def write_pose_visualisation(self, pose, path):
        html = self.get_html(pose)
        self.write_html(html, path)

    def get_html(self, pose):
        return self.get_html_header() + self.get_html_body(pose) + self.get_html_footer()

    
    def get_html_header(self):
        return visualisation_header


    def get_html_body(self, pose):
        return 

    
    def get_html_footer(self):
        if self.target == "sars2":
            return colour_sars2 + orient_tail_sars2
        elif self.target == "mers":
            return colour_mers + orient_tail_mers
        elif self.target == "7ene":
            return colour_7ene + orient_tail_7ene

