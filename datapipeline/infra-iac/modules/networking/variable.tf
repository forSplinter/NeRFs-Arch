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
