resource "aws_s3_bucket" "nerfarc_s3_bucket_default" {
    bucket = var.nerfarc_dataset_bucket
    tags = {
      Name = "${var.env}_${var.nerfarc_dataset_bucket}"
      env = var.env 
    }
  
}

resource "aws_s3_object" "nerfarc_s3_bucket_upload" {
    for_each = toset([
        "raw/uploads/",
        "raw/temp/",
        "processing/colmap-jobs/",
        "processing/augmentation/",
        "outputs/datasets/",
        "outputs/versions/"

    ]) 
    bucket = aws_s3_bucket.nerfarc_s3_bucket_default.id
    key = each.value
    server_side_encryption = "AES256"
    tags = {
      Name = "${var.env}_${var.nerfarc_dataset_bucket}"
      env = var.env

    }
  
}

resource "aws_s3_bucket_public_access_block" "diseable_public_access" {
    bucket = aws_s3_bucket.nerfarc_s3_bucket_default.id

    block_public_acls = false
    block_public_policy = false 
    ignore_public_acls = false
    restrict_public_buckets = false
  
}