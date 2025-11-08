output "nerfarc_vpc" {
    value = aws_vpc.nerfarc_vpc.id
}

output "nerfarc_public_subnet" {
    value = aws_subnet.nerfarc_public_subnet.*.id
}

output "nerfarc_private_subnet" {
    value = aws_subnet.nerfarc_private_subnet.*.id
  
}