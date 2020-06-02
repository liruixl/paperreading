## **前言**

在没有配置ssh的情况下使用ssh连接操作github库的时候会出现如下异常：

```
$ git clone git@github.com:linmuxi/test-git.git
Cloning into 'test-git'...
Warning: Permanently added the RSA host key for IP address '192.30.252.129' to the list of known hosts.
Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
~
```

## **步骤**

> 前提是我们已经新建好了一个库test-git,ssh路径是：git@github.com:lirui/test-git.git

### **1、检查ssh keys是否存在**

```
$ ls -al ~/.ssh
```

如果目录下面没有id_rsa、id_rsa.pub则表示key不存在

### **2、生成ssh key**

```
$ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
Generating public/private rsa key pair.
Enter file in which to save the key (/c/Users/Hunter/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /c/Users/Hunter/.ssh/id_rsa.
Your public key has been saved in /c/Users/Hunter/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:7KwlOZ4yljBZE2ZJ7dr8QGIyQeiPk49L+01fnC0hAZY your_email@example.com
The key's randomart image is:
+---[RSA 4096]----+
| o...=.          |
|. . *Eo          |
|.  + o .         |
| .o = o..        |
|  +* B .S.       |
| ++.. ++o +      |
| .+o o+o+= .     |
|....B..*o .      |
| ooo ++.         |
+----[SHA256]-----+
```

### **3、将ssh key添加到ssh-agent**

先确认ssh-agent是可用的

```
$ eval $(ssh-agent -s)
Agent pid 20632
```

将ssh key添加到ssh-agent

```
$ ssh-add ~/.ssh/id_rsa
Identity added: /c/Users/Hunter/.ssh/id_rsa (/c/Users/Hunter/.ssh/id_rsa)
```

### **4、将ssh key配置到github**

复制key内容

```
$ clip < ~/.ssh/id_rsa.pub
```

配置key到github
登录github->选择Settings->SSH keys->New SSH key

测试ssh key的配置情况

```
$ ssh -t git@github.com
Warning: Permanently added the RSA host key for IP address '192.30.252.128' to the list of known hosts.
PTY allocation request failed on channel 0
```

到这里就配置好了

再次执行clone操作：

```
$ git clone git@github.com:linmuxi/test-git.git
Cloning into 'test-git'...
remote: Counting objects: 56, done.
remote: Compressing objects: 100% (34/34), done.
remote: Total 56 (delta 4), reused 0 (delta 0), pack-reused 8
Receiving objects: 100% (56/56), 5.42 KiB | 0 bytes/s, done.
Resolving deltas: 100% (4/4), done.
Checking connectivity... done.
```


## SSH原理

[SSH原理与运用（一）：远程登录](http://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)

github主要使用了SSH的公钥登陆来验证身份：

使用密码登录，每次都必须输入密码，非常麻烦。好在SSH还提供了公钥登录，可以省去输入密码的步骤。

所谓"公钥登录"，原理很简单，就是用户将自己的公钥储存在远程主机上。登录的时候，远程主机会向用户发送一段随机字符串，用户用自己的私钥加密后，再发回来。远程主机用事先储存的公钥进行解密，如果成功，就证明用户是可信的，直接允许登录shell，不再要求密码。

