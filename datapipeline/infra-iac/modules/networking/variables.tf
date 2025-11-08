variable "vpc_name" {
  type = string
}

variable "vpc_cidr" {
  type = string
}

variable "vpc_cidr_public" {
    type = list(string) 
}

variable "vpc_cidr_private" {
  type = list(string)
  
}
variable "ap_available_zone" {
 
}
variable "env" {
}
