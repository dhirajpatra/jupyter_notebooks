create database practice;

create table practice.student(
id int,
firstname varchar(30),
deptid int
);

create table practice.department(
deptid int,
deptname varchar(30)
);

INSERT INTO practice.student VALUES (1,'Aman',1);
INSERT INTO practice.student VALUES (2,'Kumar',2);
INSERT INTO practice.student VALUES (3,'John',3);
INSERT INTO practice.student VALUES (4,'Doe',4);
INSERT INTO practice.student VALUES (5,'David',1);

select * from practice.student;

INSERT INTO practice.department values (1, 'IT');
INSERT INTO practice.department values (2, 'CS');
INSERT INTO practice.department values (3, 'Mechanical');
INSERT INTO practice.department values (4, 'Electrical');
INSERT INTO practice.department values (6, 'Civil');

select * from practice.department;

select * from practice.student Student left outer join practice.department department on Student.deptid = department.deptid;