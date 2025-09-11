# windows command using azcopy to download all .dcm files from the standford data, exclude teh .npy sinograms:
# azcopy copy "stanford_data_link" "target_download_path" --recursive=true --include-pattern="*.dcm"
azcopy copy "https://aimistanforddatasets01.blob.core.windows.net/ctsinogram?sv=2019-02-02&sr=c&sig=PKqiCpV3CDYgb6DfKdmizD6rzhdPpPymss75qnbtekg%3D&st=2025-09-11T13%3A57%3A25Z&se=2025-10-11T14%3A02%3A25Z&sp=rl" "the\path\you\want\to\download\to" --recursive=true --include-pattern="*.dcm"
