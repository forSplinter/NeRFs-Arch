variable "vpc_cidr" {
}
variable "vpc_name" {
}
variable "env" {
}
variable "vpc_cidr_public" {
}
variable "vpc_cidr_private" {
}
variable "availability_zone" {
}


#Vpc 
resource "aws_vpc" "nerfarc_vpc" {
    cidr_block = var.vpc_cidr 
    enable_dns_hostnames = true
    enable_dns_support = true
    tags = {
      env = var.env
      Name = "${var.env}_${var.vpc_name}"

    }
  
}

#Internal gateway 
resource "aws_internet_gateway" "nerfarc_igw" {
    vpc_id = aws_vpc.nerfarc_vpc.id
    tags = {
      env = var.env
      Name = "${var.env}_nerfarc_igw"
    }
  
}
#public subnet 
resource "aws_subnet" "nerfarc_public_subnet" {
    count = length(var.vpc_cidr_public)
    vpc_id = aws_vpc.nerfarc_vpc.id
    cidr_block = element(var.vpc_cidr_public, count.index)
    availability_zone = element(var.availability_zone, count.index)
    tags = {
      env = var.env 
      Name = "${var.env}_nerfarc_public_subnet"
    }
  
}

#private subnet 
resource "aws_subnet" "nerfarc_private_subnet" {
    count = length(var.vpc_cidr_private)
    vpc_id = aws_vpc.nerfarc_vpc.id
    cidr_block = element(var.vpc_cidr_private, count.index)
    availability_zone = element(var.availability_zone, count.index)
    tags = {
      env = var.env 
      Name = "${var.env}_nerfarc_private_subnet_${count.index + 1}"
    }
  
}

# Public route table 
resource "aws_route_table" "nerfarc_public_route_table" {
    vpc_id = aws_vpc.nerfarc_vpc.id

    route {
        cidr_block = "0.0.0.0/0"
        gateway_id = aws_internet_gateway.nerfarc_igw.id
    }

    tags = {
        Name = "${var.env}_nerfarc_route_table"
        env = var.env
    }
  
}

resource "aws_route_table_association" "nerfarc_rt_public_subnet_association" {
    count = length(aws_subnet.nerfarc_public_subnet)
    subnet_id = aws_subnet.nerfarc_public_subnet[count.index].id
    route_table_id = aws_route_table.nerfarc_public_route_table.id
}

resource "aws_route_table" "nerfarc_private_route_table" {
    vpc_id = aws_vpc.nerfarc_vpc.id
    tags = {
      Name = "${var.env}_nerfarc_private_route_table"
      env  = var.env
    }
}
resource "aws_route_table_association" "nerfarc_rt_private_subnet_association" {
    count = length(aws_subnet.nerfarc_private_subnet)
    subnet_id = aws_subnet.nerfarc_private_subnet[count.index].id
    route_table_id = aws_route_table.nerfarc_private_route_table.id
}

#security group