# windows command using azcopy to download all .dcm files from the standford data, exclude the .npy sinograms:
azcopy copy "stanford_data_link" "target_download_path" --recursive=true --include-pattern="*.dcm"
