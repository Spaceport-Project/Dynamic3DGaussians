find ./  -type f -name "Cam_*.png" | sort -V |  awk '{n=split($1,A,"_"); print A[n-1]}' | sort -V  | uniq -c   |more

colmap model_aligner   --input_path  /home/hamit/DATA/salon_photos/sparse/  --output_path  /home/hamit/DATA/salon_photos/sparse_aligned  --ref_images_path  /home/hamit/DATA/salon_photos/img_ref_1.txt  --ref_is_gps 0 --alignment_max_error 3.0

sudo certbot certonly --manual --preferred-challenges=dns --email hamit@antmedia.io  --server https://acme-v02.api.letsencrypt.org/directory --agree-tos  -d *.spaceport360.site  -d spaceport360.site

# GPU Reset
sudo fuser --kill /dev/nvidia-uvm
sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm