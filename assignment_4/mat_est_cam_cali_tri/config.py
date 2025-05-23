import logger


# directories
data_dir = "data"
lib_dir = data_dir + "/library"
lab_dir = data_dir + "/lab"
gaudi_dir = data_dir + "/gaudi"
house_dir = data_dir + "/house"
filepaths = {"lab": [f"{lab_dir}/lab1.jpg", f"{lab_dir}/lab2.jpg", f"{lab_dir}/lab_matches.txt",
                     f"{lab_dir}/lab_3d.txt"],
             "lib": [f"{lib_dir}/library1.jpg", f"{lib_dir}/library2.jpg", f"{lib_dir}/library_matches.txt",
                     f"{lib_dir}/library1_camera.txt", f"{lib_dir}/library2_camera.txt"],
             "gaudi": [f"{gaudi_dir}/gaudi1.jpg", f"{gaudi_dir}/gaudi2.jpg"],
             "house": [f"{house_dir}/house1.jpg", f"{house_dir}/house2.jpg"]}
res_dir = "results"


# ransac
# gaudi ransac hyperparameters
# dist_thres = 1.38e4
# ransac_num_iters = 50000
# ransac_thres = 1.0

# house ransac hyperparameters
# dist_thres = 3.5e4
# ransac_num_iters = 50000
# ransac_thres = 0.75


# test
plt_inliner_matches = True
imwrite = True
log_level = logger.LogLevel.INFO
