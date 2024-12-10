import mysql.connector
db=mysql.connector.connect(user="root", password="Thang2207@", host="localhost")\

#Query
code = 'CREATE SCHEMA face'

#Create table
table = 'CREATE TABLE `face`.`user` (`id` INT NOT NULL,`name` VARCHAR(45) NOT NULL,`geo_features` JSON NOT NULL,`appear_features` JSON NOT NULL,`deep_embedd` JSON NOT NULL,`image_path` MEDIUMTEXT NOT NULL,`date` DATE NOT NULL,PRIMARY KEY (`id`),UNIQUE INDEX `id_USER_UNIQUE` (`id` ASC) VISIBLE);'

#Insert
insert = 'INSERT INTO `face` VALUES (`1`, `Thang`, ``, ``, ``, ``, `` )'

#Run
mycursor=db.cursor()
mycursor.execute(code)

#update table
db.commit()