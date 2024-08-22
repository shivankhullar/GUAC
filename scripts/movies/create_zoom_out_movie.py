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
#%matplotlib inline

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



def get_full_fire_sf_snap(cloud_num, cloud_centers, gas_pos, gas_masses, gas_hsmls, pdata, refine_data):
    ### gas_pos, gas_masses, gas_hsmls are the gas particles in the fire snapshot

    #cloud_num = 11
    cloud_center = cloud_centers[cloud_num] + refine_data['Coordinates'][0] - pdata['BoxSize']/2

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

    return final_gas_pos, final_gas_masses, final_gas_hsmls, refine_coords, cloud_center






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






def plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file):
    skip_gas = False

    image_box_size = box_size
    dists_from_refine = np.sqrt((final_gas_pos[:,0] - refine_coords[0])**2 + \
                                (final_gas_pos[:,1] - refine_coords[1])**2 + \
                                (final_gas_pos[:,2] - refine_coords[2])**2)
    dist_in_kpc = image_box_size*1.25 #0.3
    inds = np.where(dists_from_refine<=dist_in_kpc)[0]

    pos, mass, hsml = final_gas_pos, final_gas_masses, final_gas_hsmls
        

    if len(inds)!=0:
        M = Meshoid(pos[inds], mass[inds], hsml[inds])
    else:
        skip_gas = True

    
    min_pos = center-image_box_size/2
    max_pos = center+image_box_size/2
    #box_size = max_pos-min_pos
    X = np.linspace(min_pos[0], max_pos[0], res)
    Y = np.linspace(min_pos[1], max_pos[1], res)

    X, Y = np.meshgrid(X, Y, indexing='ij')
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,center=center,\
                                            size=image_box_size,res=res)*1e4 #/1e6 #*1e4

    if skip_gas:
        sigma_gas_msun_pc2 = np.zeros((res,res))
    else:
        p = ax.pcolormesh(X[:, int(3.5*120):int(12.5*120)], Y[:, int(3.5*120):int(12.5*120)], sigma_gas_msun_pc2[:, int(3.5*120):int(12.5*120)], norm=colors.LogNorm(vmin=5e-1,vmax=5e3), cmap=cmap)

    if image_box_size<9:
        if stardata:
            #print ("Plotting sinks...")
            ax.scatter(stardata['Coordinates'][:,0]+cloud_center[0]-pdata['BoxSize']/2, \
                        stardata['Coordinates'][:,1]+cloud_center[1]-pdata['BoxSize']/2, c='w', s=0.1/image_box_size + np.log10(stardata['Masses']*1e14))#, alpha=star_alpha)
        if fire_stardata:
            fire_masses = fire_stardata['Masses']*1e10
            fire_masses[fire_masses<=0] = 1e-12
            ax.scatter(fire_stardata['Coordinates'][:,0]+cloud_center[0]-pdata['BoxSize']/2, \
                        fire_stardata['Coordinates'][:,1]+cloud_center[1]-pdata['BoxSize']/2, c='w', s=0.1*np.log10(fire_masses*1e11))#, alpha=star_alpha)
            

    
    ax.set_xlim([min_pos[0], max_pos[0]])
    ax.set_ylim([min_pos[1]+(3.5*box_size/16), max_pos[1]-(3.5*box_size/16)])


    ax.minorticks_on()
    ax.tick_params(which = 'both', direction = 'in', \
                    right = True, top=True, labelsize=0)
    ax.tick_params(which='major', length=0, width=0)
    ax.tick_params(which='minor', length=0, width=0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])


    fontprops = fm.FontProperties(size=20)
    

    scale_bar_size = int(image_box_size/3)//10*10
    fontprops = fm.FontProperties(size=20)

    print (image_box_size, scale_bar_size)
    if scale_bar_size>=1000:
        scale_bar_size=1000
        scale_text = '{scale} Mpc'.format(scale=scale_bar_size//1000)
    elif scale_bar_size>=10 and scale_bar_size<1000:
        scale_text = '{scale} kpc'.format(scale=scale_bar_size)
    elif scale_bar_size<10 and scale_bar_size>=1:
        scale_bar_size = int(image_box_size/3)
        scale_text = '{scale} kpc'.format(scale=scale_bar_size)
    elif scale_bar_size<1:
        scale_bar_size = int(image_box_size/3)
        scale_text = '{scale} kpc'.format(scale=scale_bar_size)
        box_size_in_pc = image_box_size*1000
        if box_size_in_pc<3000:
            scale_bar_size_in_pc = int(box_size_in_pc/3)
            scale_text = '{scale} pc'.format(scale=scale_bar_size_in_pc)
            scale_bar_size = scale_bar_size_in_pc/1000

    scalebar = AnchoredSizeBar(ax.transData,
                            scale_bar_size, scale_text, 'upper left', 
                            pad=1,
                            color='white',
                            frameon=False,
                            size_vertical=scale_bar_size/100, 
                            fontproperties=fontprops)


    ax.add_artist(scalebar)

    plt.tight_layout() 
    #plt.show()
    plt.savefig(save_file, dpi=100)#, bbox_inches='tight', pad_inches=0)
    plt.close()















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

save_path = '../../../SFIRE/final_images/'
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





# Let's start by defining when to zoom in, when to stop and run, when to stop and zoom out and when to stop and finish
begin_sim_snap = 692
begin_zoom_in_snap = 692
begin_zoom_in_box_size = 0.1
end_zoom_in_box_size = 0.01

begin_first_zoom_out_snap = 1071
begin_first_zoom_out_box_size = 0.01
end_first_zoom_out_box_size = 0.05

begin_second_zoom_out_snap = 1195
begin_second_zoom_out_box_size = 0.05
end_second_zoom_out_box_size = 0.3

end_sim_snap = 1332
begin_final_zoom_out_box_size = 0.3
end_final_zoom_out_box_size = 200
snapshot_suffix=''

# First zoom in
snap = begin_second_zoom_out_snap #begin_sim_snap
pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)

cloud_num = 11
final_gas_pos, final_gas_masses, final_gas_hsmls, refine_coords, cloud_center = get_full_fire_sf_snap(cloud_num, cloud_centers, gas_pos, gas_masses, gas_hsmls, pdata, refine_data)

center = refine_coords 
cmap = cc.cm['fire']
#cmap = 'inferno'
res = 1920 #1024


"""
first_zoom_in_box_sizes = np.logspace(np.log10(begin_zoom_in_box_size), np.log10(end_zoom_in_box_size), 200)
print (len(first_zoom_in_box_sizes))

count = 0
for box_size in first_zoom_in_box_sizes:
    print (count)
    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)
    count = count+1


print ("-------------xxxx----- Done with phase 1 -----xxxx------------")

count = 0
first_run_snaps = np.arange(begin_sim_snap+1, begin_first_zoom_out_snap, 1)
print (len(first_run_snaps))
for snap in first_run_snaps:
    pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)
    time = pdata['Time']*kpc/1e2/Myr
    print ("Time:" , time, snap)

    final_gas_pos, final_gas_masses, final_gas_hsmls, refine_coords, cloud_center = get_full_fire_sf_snap(cloud_num, cloud_centers, gas_pos, gas_masses, gas_hsmls, pdata, refine_data)

    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)
    #count = count+1


print ("-------------xxxx----- Done with phase 2 -----xxxx------------")


# First zoom out
snap = begin_first_zoom_out_snap  #1102
snapshot_suffix=''
pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)

first_zoom_out_box_sizes = np.logspace(np.log10(begin_first_zoom_out_box_size), np.log10(end_first_zoom_out_box_size), 200)
print (len(first_zoom_out_box_sizes))

count = 0
for box_size in first_zoom_out_box_sizes:
    print (count)
    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)
    count = count+1


print ("-------------xxxx----- Done with phase 3 -----xxxx------------")


count = 0
second_run_snaps = np.arange(begin_first_zoom_out_snap+1, begin_second_zoom_out_snap, 1)
print (len(second_run_snaps))

for snap in second_run_snaps:
    pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)
    time = pdata['Time']*kpc/1e2/Myr
    print ("Time:" , time, snap)

    final_gas_pos, final_gas_masses, final_gas_hsmls, refine_coords, cloud_center = get_full_fire_sf_snap(cloud_num, cloud_centers, gas_pos, gas_masses, gas_hsmls, pdata, refine_data)

    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)
    

print ("-------------xxxx----- Done with phase 4 -----xxxx------------")
"""
 


# Second zoom out
snap = begin_second_zoom_out_snap  #1228
snapshot_suffix=''
pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)

second_zoom_out_box_sizes = np.logspace(np.log10(begin_second_zoom_out_box_size), np.log10(end_second_zoom_out_box_size), 200)

count = 0
for box_size in second_zoom_out_box_sizes:
    print (count)
    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)
    count = count+1



print ("-------------xxxx----- Done with phase 5 -----xxxx------------")



count = 0

third_run_snaps = np.arange(begin_second_zoom_out_snap+1, end_sim_snap, 1)
print (len(third_run_snaps))
       
for snap in third_run_snaps:
    pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)
    time = pdata['Time']*kpc/1e2/Myr
    print ("Time:" , time, snap)

    final_gas_pos, final_gas_masses, final_gas_hsmls, refine_coords, cloud_center = get_full_fire_sf_snap(cloud_num, cloud_centers, gas_pos, gas_masses, gas_hsmls, pdata, refine_data)
    
    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)


print ("-------------xxxx----- Done with phase 6 -----xxxx------------")



# Final zoom out
snap = end_sim_snap  #1332
snapshot_suffix=''
pdata, stardata, fire_stardata, refine_data, snapname = get_snap_data_plotting(sim, sim_path, snap, snapshot_suffix=snapshot_suffix)

final_zoom_out_box_sizes = np.logspace(np.log10(begin_final_zoom_out_box_size), np.log10(end_final_zoom_out_box_size), 300)
print (len(final_zoom_out_box_sizes))

count = 0
for box_size in final_zoom_out_box_sizes:
    print (count)
    save_file_name = 'movie_{snap}_{count}.png'.format(snap=snap, count=count)
    save_file = save_path+save_file_name
    plot_box(final_gas_pos, final_gas_masses, final_gas_hsmls, stardata, fire_stardata, box_size, center, cloud_center, save_file)
    count = count+1


print ("-------------xxxx----- Done with phase 7 -----xxxx------------")