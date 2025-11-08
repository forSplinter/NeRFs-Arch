variable "region" {
  type = string
  default = "ap-southeast-2"
  description = "aws location"
}

variable "env" {
    type = string
    description = "value"
}

variable "vpc_name" {
  type = string
  description = "value"
}

variable "vpc_cidr" {
  type = string
  description = "value"
}

variable "vpc_cidr_public" {
    type = list(string) 
    description = "value"
}

variable "vpc_cidr_private" {
  type = list(string)
  description = "value"
  
}
variable "ap_available_zone" {
  type = list(string)
  description = "value"
  
}

variable "nerfarc_dataset_bucket" {
  type = string
  description = "value"
  
}