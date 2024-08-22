
import sys
sys.path.insert(0, '../src/')
#from IC_funcs import *
from epsff_calc import *
from viz_utils import *
from cloud_selector_funcs import *

from fire_utils import *

from meshoid import Meshoid

from matplotlib.colors import LogNorm
from matplotlib import colors

import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from cloud_analysis import *
#matplotlib.use('Agg')
import h5py

import colorcet as cc
from matplotlib.cm import get_cmap

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})





def get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=''):
    if snap<10:
        snapname = 'snapshot_'+snapshot_suffix+'00{num}'.format(num=snap) 
    elif snap>=10 and snap<100:
        snapname = 'snapshot_'+snapshot_suffix+'0{num}'.format(num=snap)
    else:
        snapname = 'snapshot_'+snapshot_suffix+'{num}'.format(num=snap) 
            
    F = h5py.File(sim_path+sim+'/snapshots/'+snapname+'.hdf5',"r")
    #rho = F["PartType0"]["Density"][:]
    #density_cut = (rho*300 > .1)
    pdata = {}
    for field in "Masses", "Density", "Coordinates", "SmoothingLength", "Velocities", "ParticleIDGenerationNumber":
        pdata[field] = F["PartType0"][field][:]#[density_cut]

    for key in F['Header'].attrs.keys():
        pdata[key] = F['Header'].attrs[key]
    
    stardata = {}
    if 'PartType5' in F.keys():
        for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber", "StellarFormationTime":
            stardata[field] = F["PartType5"][field][:]#[density_cut]

        for key in F['Header'].attrs.keys():
            stardata[key] = F['Header'].attrs[key]

    fire_stardata = {}
    if 'PartType4' in F.keys():
        for field in "Masses", "Coordinates", "Velocities", "ParticleIDGenerationNumber":
            fire_stardata[field] = F["PartType4"][field][:]#[density_cut]

        for key in F['Header'].attrs.keys():
            fire_stardata[key] = F['Header'].attrs[key]
    
    refine_data = {}
    if 'PartType3' in F.keys():
        for field in "Masses", "Coordinates", "Velocities":
            refine_data[field] = F["PartType3"][field][:]#[density_cut]

    #refine_pos = np.array(F['PartType3/Coordinates'])
    #refine_pos = refine_pos[0]
    F.close()

    return pdata, stardata, fire_stardata, refine_data, snapname



path = "../../../FIRE-2/" #m12i_final/"

start_snap = 591
last_snap = 614
#field_names = ["names", "masses"]
filename_prefix = "Linked_Clouds_"

#No. of digits in the names of the clouds and the snapshots.
cloud_num_digits = 4
snapshot_num_digits = 4

cloud_prefix = "Cloud"
image_path = 'img_data/'
image_filename_prefix = 'image_faceon_s0' #'center_proj_'
image_filename_suffix = '_fov0035_Ngb32_star.hdf5' #'.hdf5'
hdf5_file_prefix = 'Clouds_'


age_cut = 1
dat_file_header_size=8
snapshot_prefix="Snap"
star_data_sub_dir = "StarData/"
cph_sub_dir="CloudPhinderData/"
#cph_sub_dir='m12i_restart/'
frac_thresh='thresh0.3'

r_gal = 25
h = 0.4 #0.4

save_path = '../../../SFIRE/dm_images/'
#snapnum = 650

nmin = 10
vir = 5
sim = "m12i_restart_2"
sub_dir = "CloudTrackerData/n{nmin}_alpha{vir}/".format(nmin=nmin, vir=vir)
filename = path+sub_dir+filename_prefix+str(start_snap)+"_"+str(last_snap)+"names"+".txt"

params = Params(path, nmin, vir, sub_dir, start_snap, last_snap, filename_prefix, cloud_num_digits, \
                snapshot_num_digits, cloud_prefix, snapshot_prefix, age_cut, \
                dat_file_header_size, star_data_sub_dir, cph_sub_dir,\
                image_path, image_filename_prefix,\
                image_filename_suffix, hdf5_file_prefix, frac_thresh, sim=sim, r_gal=r_gal, h=h)


snap_num = 601
snapdir = params.path+'snapdir_{num}/'.format(num=snap_num)
dm_masses = load_from_snapshot.load_from_snapshot('Masses', 1, snapdir, snap_num, \
                                                  units_to_physical=True)
dm_pos = load_from_snapshot.load_from_snapshot('Coordinates', 1, snapdir, snap_num, units_to_physical=True)


gas_masses = load_from_snapshot.load_from_snapshot('Masses', 0, snapdir, snap_num, units_to_physical=True)
gas_pos = load_from_snapshot.load_from_snapshot('Coordinates', 0, snapdir, snap_num, units_to_physical=True)
gas_hsmls = load_from_snapshot.load_from_snapshot('SmoothingLength', 0, snapdir, snap_num, units_to_physical=True)




cloud_data_path = path+sim+"/"+'CloudPhinderData/n{nmin}_alpha{vir}/'.format(nmin=nmin, vir=vir)
filename = 'bound_{snap_num}_n{nmin}_alpha{vir}.dat'.format(snap_num=snap_num, nmin = nmin, vir=vir)
datContent = [i.strip().split() for i in open(cloud_data_path+filename).readlines()]

cloud_total_mass_list = []
cloud_centre_list = []
cloud_reff_list = []
cloud_hmrad_list = []
cloud_num_part_list = []
cloud_vir_list = []
z_dists = []
r_gal_dists = []
cloud_nums = []

for i in range (0, len(datContent)):
    j = i-8
    if (i<8):
        continue
    cloud_total_mass = float(datContent[i][0])
    cloud_centre_x = float(datContent[i][1])
    cloud_centre_y = float(datContent[i][2])
    cloud_centre_z = float(datContent[i][3])
    cloud_reff = float(datContent[i][7])
    cloud_hmrad = float(datContent[i][8])
    cloud_num_part = float(datContent[i][9])
    cloud_vir = float(datContent[i][10])
    
    cloud_total_mass_list.append(cloud_total_mass)
    cloud_centre_list.append([cloud_centre_x, cloud_centre_y, cloud_centre_z])
    cloud_reff_list.append(cloud_reff)


cloud_centers = np.array(cloud_centre_list)


sim = 'Cloud0011Snap601_original_early5_sf_refine'
sim_path = '../../../SFIRE/'

# We will use the final snapshot to zoom out fully...
snap = 1332  #716 #832
snapshot_suffix=''
pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)


cloud_num = 11
cloud_center = cloud_centers[cloud_num] + refine_data['Coordinates'][0] - pdata['BoxSize']/2
#print (cloud_center)

# Remove the particles that are in the box from gas_pos, gas_masses and gas_hsmls and then add the cloud particles from pdata['Coordinates'], pdata['Masses'] and pdata['SmoothingLength']
# to gas_pos, gas_masses and gas_hsmls
box_size = pdata['BoxSize']
box_inds = np.where((gas_pos[:,0] > cloud_center[0]-box_size/2) & (gas_pos[:,0] < cloud_center[0]+box_size/2) & \
                (gas_pos[:,1] > cloud_center[1]-box_size/2) & (gas_pos[:,1] < cloud_center[1]+box_size/2) & \
                (gas_pos[:,2] > cloud_center[2]-box_size/2) & (gas_pos[:,2] < cloud_center[2]+box_size/2))[0]

final_gas_pos = np.delete(gas_pos, box_inds, axis=0)
final_gas_masses = np.delete(gas_masses, box_inds)
final_gas_hsmls = np.delete(gas_hsmls, box_inds)

refined_cloud_coords_x = pdata['Coordinates'][:,0] + cloud_center[0] - pdata['BoxSize']/2
refined_cloud_coords_y = pdata['Coordinates'][:,1] + cloud_center[1] - pdata['BoxSize']/2
refined_cloud_coords_z = pdata['Coordinates'][:,2] + cloud_center[2] - pdata['BoxSize']/2
refined_cloud_coords = np.array([refined_cloud_coords_x, refined_cloud_coords_y, refined_cloud_coords_z]).T

final_gas_pos = np.concatenate((final_gas_pos, refined_cloud_coords), axis=0)
final_gas_masses = np.concatenate((final_gas_masses, pdata['Masses']))
final_gas_hsmls = np.concatenate((final_gas_hsmls, pdata['SmoothingLength']))

refine_coords = refine_data['Coordinates'][0] - pdata['BoxSize']/2 + cloud_center




pos, mass = dm_pos, dm_masses
hsml = np.ones(len(mass))*0.3
M_dm = Meshoid(pos, mass, hsml)

pos, mass, hsml = final_gas_pos, final_gas_masses, final_gas_hsmls
M_gas = Meshoid(pos, mass, hsml)



initial_dm_alpha=0
initial_gas_alpha=1
final_gas_alpha=0
final_dm_alpha=1
# We will smoothly transition between the two after getting to a box size of 500
# The transition will happen over 50 frames


# Deal with slowing down the zoom in once we reach certain box sizes. Just give a list of box_breaks
#box_breaks = [1000,200]
# Decide what percentage of the total time to be spent on the given box sizes
#box_size_time_percentages = [0.3, 0.3, 0.4] #should sum to 1 and be 1 more in length than box_breaks

# Make an array of the box_sizes
times = np.arange(0, 800)

initial_box_size = 200 #4000
final_box_size = 4000

box_breaks = [400, 1000]
#box_breaks = [800,100]
box_size_time_percentages = [0.4, 0.3, 0.3] #should sum to 1 and be 1 more in length than box_breaks
box_sizes = np.linspace(initial_box_size, box_breaks[0], int(box_size_time_percentages[0]*len(times)))
for i in range(1, len(box_breaks)):
    #print (np.linspace(box_sizes[-1], box_breaks[i], int(box_size_time_percentages[i]*len(times))))
    box_sizes = np.append(box_sizes[:-1], np.linspace(box_sizes[-1], box_breaks[i], int(box_size_time_percentages[i]*len(times))))

box_sizes = np.append(box_sizes[:-1], np.linspace(box_sizes[-1], final_box_size, int(box_size_time_percentages[-1]*len(times))))
box_sizes = np.append(box_sizes, np.ones(len(times)-len(box_sizes))*final_box_size)

print (box_sizes)




res = 1920 #change to 2048 later maybe

center = refine_coords #gal_center_proj

transition_times = 30
gas_transition_times = 30
count = 0
#gas_count = 471
for i in range(471, len(times)):
    time = i
    current_box_size = box_sizes[i] #initial_box_size - time*(initial_box_size-final_box_size)/len(times)
    
    image_box_size = current_box_size
    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2

    if current_box_size > 800:
        current_dm_alpha = initial_dm_alpha + count*(final_dm_alpha - initial_dm_alpha)/transition_times
        current_gas_alpha = initial_gas_alpha - count*(initial_gas_alpha-final_gas_alpha)/transition_times
        count += 1
        if current_dm_alpha < 0:
            current_dm_alpha = 0
        if current_gas_alpha > 1:
            current_gas_alpha = 1
        
    else:
        current_dm_alpha = 0
        current_gas_alpha = 1

    print (time, current_box_size, current_dm_alpha, current_gas_alpha)

    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)
    X, Y = np.meshgrid(X, Y, indexing='ij')

    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    if current_dm_alpha>0:
        if current_dm_alpha>1:
            current_dm_alpha=1

        sigma_dm_msun_pc2 = M_dm.SurfaceDensity(M_dm.m*1e10,center=center,\
                                                size=image_box_size,res=res) #*1e4
        sigma_dm_msun_pc2[sigma_dm_msun_pc2<1e3] = 1e3
        #p2 = ax.pcolormesh(X[:, int(3.5*120):int(12.5*120)], Y[:, int(3.5*120):int(12.5*120)], sigma_dm_msun_pc2[:, int(3.5*120):int(12.5*120)], norm=colors.LogNorm(vmin=1e3,vmax=1e9), cmap='inferno', alpha=current_dm_alpha)
        ax.imshow(sigma_dm_msun_pc2[:, int(3.5*120):int(12.5*120)].T, norm=colors.LogNorm(vmin=1e3,vmax=1e9), cmap='inferno', alpha=current_dm_alpha, origin='lower')


    if current_gas_alpha > 0:
        if current_gas_alpha < 0:
            current_gas_alpha = 0
        if current_gas_alpha > 1:
            current_gas_alpha = 1

        sigma_gas_msun_pc2 = M_gas.SurfaceDensity(M_gas.m*1e10,center=center,\
                                            size=image_box_size,res=res)/1e6 #*1e4
        #sigma_gas_msun_pc2[sigma_gas_msun_pc2<1] = 1
        ax.imshow(sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)].T, norm=colors.LogNorm(vmin=5e-1,vmax=5e3), cmap='cet_fire', alpha=current_gas_alpha, origin='lower')

    
        #
        #if current_box_size>500:
        #    current_gas_alpha1 = initial_dm_alpha + gas_count*(final_dm_alpha - initial_dm_alpha)/gas_transition_times
        #    current_gas_alpha2 = initial_gas_alpha - gas_count*(initial_gas_alpha - final_gas_alpha)/gas_transition_times
        #    if current_gas_alpha1 < 0:
        #        current_gas_alpha1 = 0
        #    if current_gas_alpha2 > 1:
        #        current_gas_alpha2 = 1

        #    print (current_gas_alpha1, current_gas_alpha2, gas_count)
        #    gas_count += 1
        #    if current_gas_alpha1 < 0:
        #        current_gas_alpha1 = 0
        #    if current_gas_alpha2 > 1:
        #        current_gas_alpha2 = 1
        #    #p2 = ax.pcolormesh(X[:, int(3.5*120):int(12.5*120)], Y[:, int(3.5*120):int(12.5*120)], sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)], norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap='inferno', alpha=current_gas_alpha1)
        #    print (time, current_box_size, current_dm_alpha, current_gas_alpha)
        #    ax.imshow(sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)].T, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap='inferno', alpha=current_gas_alpha1, origin='lower')

        #    #p = ax.pcolormesh(X[:, int(3.5*120):int(12.5*120)], Y[:, int(3.5*120):int(12.5*120)], sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)], norm=colors.LogNorm(vmin=5e-1,vmax=5e3), cmap='cet_fire', alpha=current_gas_alpha2)
        #ax.imshow(sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)].T, norm=colors.LogNorm(vmin=5e-1,vmax=5e3), cmap='cet_fire', alpha=current_gas_alpha, origin='lower')

        #else:
        #    #p = ax.pcolormesh(X[:, int(3.5*120):int(12.5*120)], Y[:, int(3.5*120):int(12.5*120)], sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)], norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap='inferno', alpha=current_gas_alpha)
        #    ax.imshow(sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)].T, norm=colors.LogNorm(vmin=1e-4,vmax=1), cmap='inferno', alpha=current_gas_alpha, origin='lower')

        
    #ax.set_aspect('equal')
    
    #ax.set_xlim([min_pos[0], max_pos[0]])
    #ax.set_ylim([min_pos[1]+(3.5*box_size/16), max_pos[1]-(3.5*box_size/16)])
    ax.set_xticks([])
    ax.set_yticks([])



    scale_bar_size = int(image_box_size/3)//10*10
    fontprops = fm.FontProperties(size=20)
    if scale_bar_size>=1000:
        scale_bar_size=1000
        scale_text = '{scale} Mpc'.format(scale=scale_bar_size//1000)
    elif scale_bar_size>=10 and scale_bar_size<1000:
        scale_text = '{scale} kpc'.format(scale=scale_bar_size)
    elif scale_bar_size<10:
        scale_bar_size = int(image_box_size/3)
        scale_text = '{scale} kpc'.format(scale=scale_bar_size)

    scalebar = AnchoredSizeBar(ax.transData,
                            scale_bar_size*res/image_box_size, scale_text, 'upper left', 
                            pad=1,
                            color='white',
                            frameon=False,
                            size_vertical=scale_bar_size/100*res/image_box_size, 
                            fontproperties=fontprops)

    ax.add_artist(scalebar)

    #plt.colorbar(p, label='Surface Density [M$_\odot$/pc$^2$]')
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path+'/dm_movie_{time}.png'.format(time=time), dpi=100)
    plt.close()
