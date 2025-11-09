resource "aws_s3_bucket" "nerfarc_s3_bucket_default" {
    bucket = var.nerfarc_dataset_bucket
    tags = {
      Name = "${var.env}_${var.nerfarc_dataset_bucket}"
      env = var.env 
    }
  
}

resource "aws_caller_identity" "name" {
  
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

    block_public_acls = true
    block_public_policy = true 
    ignore_public_acls = true
    restrict_public_buckets = true 
  
}

resource "aws_s3_bucket_policy" "nerfarc_s3_bucket_policy" {
  bucket = aws_s3_bucket.nerfarc_s3_bucket_default.id
  policy =  jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" //TODO: Refacto the arn role need to be the lambda here
        Action = [
          "s3:ListBucket",
          "s3:PutObject",
          "s3:GetObject",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.nerfarc_s3_bucket_default.arn,
          "${aws_s3_bucket.nerfarc_s3_bucket_default.arn}*//*"
        ]
      }
    ]
  }) 
  depends_on = [aws_s3_bucket_public_access_block.diseable_public_access]
}