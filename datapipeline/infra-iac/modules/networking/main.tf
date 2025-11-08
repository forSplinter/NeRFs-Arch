variable "vpc_cidr" {
}
variable "vpc_name" {
}
variable "env" {
}


#Vpc 
resource "aws_vpc" "nerfarc_vpc" {
    cidr_block = var.vpc_cidr 
    enable_dns_hostnames = true
    enable_dns_support = true
    tags = {
      Name = var.vpc_name
    }
  
}

#Internal gateway 
resource "aws_internet_gateway" "nerfarc_igw" {
    vpc_id = aws_vpc.main.id
    tags = {
      env = var.env
      Name = "${var.env}_nerfarc_igw"
    }
  
}
#public subnet 
resource "aws_subnet" "nerfarc_public_subnet" {
    vpc_id = aws_vpc.prod_nerfarc_vpc.id
    cidr_block = var.vpc_cidr
    tags = {
      env = var.env 
      Name = "${var.env}_nerfarc_public_subnet"
    }
  
}