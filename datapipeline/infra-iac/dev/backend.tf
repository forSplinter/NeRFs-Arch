terraform {
  backend "s3" {
    bucket = "aws-nerfarc-s3"
    key = "aws-nerfarc-s3/metadata/terraform.tfstate"
    region = "ap-southeast-2"
  }
}