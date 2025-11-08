variable "region" {
  type = string
  default = "us-east-1"
  description = "aws location"
}

variable "storage_principal_name" {
    type = string
}

variable "bucket_name" {
  type = string
  description = "value"
}

variable "vpc_cidr" {
    type = string
    description = "value"
}

variable "vpc_name" {
  type = string
  description = "value"
}

variable "env" {
    type = string
    description = "value"
}