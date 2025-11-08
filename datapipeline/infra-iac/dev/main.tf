module "networking" {
    source = "../modules/networking"
    vpc_cidr = var.vpc_cidr
    vpc_cidr_public = var.vpc_cidr_public
    vpc_cidr_private = var.vpc_cidr_private
    ap_available_zone = var.ap_available_zone
    env = var.env
    vpc_name = var.vpc_name
}
