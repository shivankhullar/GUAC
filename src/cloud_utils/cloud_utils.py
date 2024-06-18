
"""
def get_cloud_name(cloud_num, params):
    
    Get the cloud name for the given cloud number.

    Inputs:
    cloud_num: int, cloud number
    params: object, parameters object

    Output:
    name: string, cloud name
    
    if cloud_num<10:
        name = params.cloud_prefix+"00"+str(cloud_num)
    if cloud_num>=10 and cloud_num<100:
        name = params.cloud_prefix+"0"+str(cloud_num)
    if cloud_num>=100 and cloud_num<1000:
        name = params.cloud_prefix+str(cloud_num)
    else:
        name = params.cloud_prefix+str(cloud_num)
    return name
"""



class CloudChain():
    """
    Class to find out a cloud chain stemming from the first cloud.
    """
    def __init__(self, cloud_num, snap_num, params):
        file_name = params.path+params.sub_dir+params.filename_prefix+params.frac_thresh\
                    +"_"+str(params.start_snap)+"_"+str(params.last_snap)+"_names"+".txt"
        my_file = open(file_name, "r")

        content_list = my_file.readlines()
        cloud_list_names = []
        for i in range (0, len(content_list)-1):             #The last line is just \n.
            #if 
            names = str.split(content_list[i], ', ')
            if names[-1]=='\n':
                names = names[:-1]

            cloud_list_names.append(names)

        self.cloud_list = []
        self.cloud_nums = []
        self.snap_nums = []
        search_key = get_cloud_name(cloud_num, params)+'Snap'+str(snap_num)
        self.search_key = search_key
        flag = 0
        for i in range(0, len(cloud_list_names)):
            if search_key in cloud_list_names[i]:
                print ('Search key', search_key, i)
                self.cloud_list = cloud_list_names[i]

                flag = 1
                break
        
        for cloud in self.cloud_list:
            self.cloud_nums.append(int(cloud.split('Snap')[0].split('Cloud')[1]))
            self.snap_nums.append(int(cloud.split('Snap')[1]))
        
        #if flag==0:
        #    search_key = get_cloud_name0(cloud_num, params)+'Snap'+str(snap_num)
        #    for i in range(0, len(cloud_list_names)):
        #        if search_key in cloud_list_names[i]:
        #            print ('Search key', search_key, i)
        #            self.cloud_list = cloud_list_names[i]
        #            flag = 1
        #            break
                    
        if flag==0:
            print ('Cloud not found :(')
            

