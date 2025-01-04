from brainrender import settings
from brainrender import Scene
#from bg_atlasapi import show_atlases
from brainrender.actors import Cylinder
from brainrender.camera import set_camera
from brainrender import VideoMaker

def save_csv_with_list_of_abbreviations(scene):
    # run this to make a csv file with all tha names of the brain areas available in the atlas
    regions = scene.atlas.lookup_df
    regions.to_csv('regions.csv')
    print(regions.head())
    print(scene.atlas.hierarchy)


def draw_brain():
    settings.SHADER_STYLE = 'plastic'  # other options: metallic, plastic, shiny, glossy, cartoon, default
    settings.ROOT_ALPHA = .1   # this sets how transparent the brain outline is
    settings.SHOW_AXES = False  # hides the axes from the image
    #show_atlases()  # this will print a list of atlases that you could use
    scene = Scene(root=False, inset=False, atlas_name="allen_mouse_10um")  # makes a scene instance using the default atlas
    regions = scene.atlas.lookup_df
    #print(regions)
    #for i in range(len(regions)):
    #    print(regions["acronym"].iloc[i], ":", regions["name"].iloc[i])
    #print(scene.atlas.hierarchy)
    root = scene.add_brain_region("root", alpha=0.1, color="grey")  # this is the brain outline
    mec = scene.add_brain_region("ENTm", alpha=0.5, color="#01665e", hemisphere=None) 
    ca1 = scene.add_brain_region("CA1", alpha=0.5, color="#8c510a", hemisphere=None)
    pre = scene.add_brain_region("PRE", alpha=0.5, color="#5ab4ac", hemisphere=None)
    par = scene.add_brain_region("PAR", alpha=0.5, color="#5ab4ac", hemisphere=None)
    post = scene.add_brain_region("POST", alpha=0.5, color="#f6e8c3", hemisphere=None)
    rsp = scene.add_brain_region("RSP", alpha=0.5, color="#d8b365", hemisphere=None)
    #vis = scene.add_brain_region("VIS", alpha=0.25, color=(67,142,137), hemisphere="right", silhouette=True)

    #scene.render(zoom=2)  # this line will display the image
    #scene.render(zoom=2, camera="sagittal")
    print("rendered")  
    #scene.export("/mnt/datastore/Harry/brain_regions.html")  
    #scene.screenshot(r"C:\Users\harry\OneDrive\Desktop\brainrender\render1.png")

    print("made html")
    # Make a custom make frame function
    def make_frame(scene, frame_number, *args, **kwargs):
        alpha = scene.root.alpha()
        if alpha < 0.5:
            scene.root.alpha(1)
        else:
            scene.root.alpha(0.2)

    # Create an instance of video maker 
    vm = VideoMaker(scene, save_fld=".", name="vid1", fmt="mp4")  
    # make a video with the custom make frame function
    # this just rotates the scene
    render_dict = {"zoom": 2,
                "camera": "sagittal"}
    vm.make_video(elevation=0, roll=0, azimuth=0.5, duration=1, fps=60, render_kwargs=render_dict)
    print("hello")

def add_actor(scene, mec):
    actor = Cylinder(mec, scene.root)
    scene.add(actor)
    return scene


def main():
    draw_brain()


if __name__ == '__main__':
    main()