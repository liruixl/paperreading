# socket 地址API

字节序问题，现代PC大多采用小端字节序，因此小端字节序又称为主机字节序。

大端字节序也称为网络字节序。JAVA虚拟机采用大端字节序。

## 字节序转化

```c++
#include <netinet/in.h>
//typedef unsigned short int	uint16_t;
//typedef unsigned int		uint32_t;

extern uint32_t ntohl (uint32_t __netlong);
extern uint16_t ntohs (uint16_t __netshort);
extern uint32_t htonl (uint32_t __hostlong);
extern uint16_t htons (uint16_t __hostshort);
```

望文生义：`htonl` 表示host to network long，即将长整型（32位）的主机字节序数据转化为网络字节序数据。

+ 长整型函数通常用来转化IP地址。
+ 短整形函数用来转化端口号。

## 通用socket地址

```c++
#inlcude <bits/socket.h>

//socketaddr.h文件里定义的宏
typedef unsigned short int sa_family_t;
#define	__SOCKADDR_COMMON(sa_prefix) \
  sa_family_t sa_prefix##family
//=====================================


/* Structure describing a generic socket address.  */
struct sockaddr
  {
    __SOCKADDR_COMMON (sa_);	/* Common data: address family and length.  */
    char sa_data[14];		/* Address data.  */
  };

struct sockaddr
  {
    sa_family_t sa_family; //地址族
    char sa_data[14]; //地址，不同地址族有不同含义
  };

//地址族，协议族，sa_data含义
#define AF_UNIX		PF_UNIX    //本地域协议族，文件路径名
#define AF_INET		PF_INET    //TCP/IPv4协议族，16bit端口号 32bit IPv4地址，共6字节
#define AF_INET6	PF_INET6   //TCP/IPv6协议族，26字节，sa_data不够了...
```

## 专用 socket 地址

通用socket地址显然很不好用。。。需要执行繁琐的位操作，Linux为各个协议族提供了专门的 socket 地址结构体。

实际使用的时候需要转化为通用的socket地址类型`sockaddr`（强制转化即可），因为所有socket编程接口使用的地址参数类型都是sockaddr。

```c++
#include <netinet/in.h>

/* Structure describing an Internet socket address.  */
typedef uint16_t in_port_t;
/* Internet address.  */
typedef uint32_t in_addr_t;
struct in_addr
  {
    in_addr_t s_addr;  //IPv4地址，需要网络字节序表示
  };

//IPv4专用socket地址结构
struct sockaddr_in
  {
    __SOCKADDR_COMMON (sin_);    //sa_family_t sa_family; 
    in_port_t sin_port;			/* Port number. . 用网络字节序表示 */
    struct in_addr sin_addr;	 /* Internet address. 见上面 */

    /* Pad to size of `struct sockaddr'.  */
    unsigned char sin_zero[sizeof (struct sockaddr) -
			   __SOCKADDR_COMMON_SIZE -
			   sizeof (in_port_t) -
			   sizeof (struct in_addr)];
  };
```

## IP地址转化（字符串 整数（二进制））

点分十进制字符串<=======>网络字节序整数表示的IPv4地址

```c++
#include <arpa/inet.h>

/* Convert Internet host address from numbers-and-dots notation in CP
   into binary data in network byte order.  */
extern in_addr_t inet_addr (const char *__cp);
/* same as inet_addr, but store the result in the structure INP.  */
extern int inet_aton (const char *__cp, struct in_addr *__inp); //成功返回1 否则0

/* Convert Internet number in IN to ASCII representation.  The return value
   is a pointer to an internal array containing the string.  */
extern char *inet_ntoa (struct in_addr __in);//静态变量存储结果，不可重入
```

使用样例：

```c++
int main()
{
    sockaddr_in addr4;

    uint32_t ip_bin =  inet_addr("192.168.1.1");
    std::cout  << "192.168.1.1 to num: " << ip_bin << std::endl;

    inet_aton("192.168.1.1", &addr4.sin_addr);
    char * ipstr = inet_ntoa(addr4.sin_addr);
    printf("ip str: %s\n", ipstr);
    
    return 0;
}
//控制台输出
// 192.168.1.1 to num: 16885952
// ip str: 192.168.1.1
```

## IP地址转换 更新函数

下面两个转化函数同时适用于IPv4地址和IPv6地址。

```c++
#include <arpa/inet.h>
/* Convert from presentation format of an Internet number in buffer
   starting at CP to the binary network format and store result for
   interface type AF in buffer starting at BUF.  */
// p 互联网号码的显示格式
// n network format
extern int inet_pton (int __af, const char * __cp, void * __buf);

/* Convert a Internet address in binary network format for interface
   type AF in buffer starting at CP to presentation form and place
   result in buffer of length LEN astarting at BUF.  */
extern const char *inet_ntop (int __af, const void * __cp, char *__buf, socklen_t __len);
```

示例：

```c++
//IPv4转化示例
sockaddr_in addr4;
inet_pton(AF_INET, "255.255.255.0", &addr4.sin_addr);
char dst[INET_ADDRSTRLEN ];
const char * client = inet_ntop(AF_INET, &addr4.sin_addr, dst, INET_ADDRSTRLEN);
printf("test num to ip: %s \n", client);
```

# socket API

```c++
#include <sys/socket.h>
```

## 创建socket

```c++
/* Create a new socket of type TYPE in domain DOMAIN, using
   protocol PROTOCOL.  If PROTOCOL is zero, one is chosen automatically.
   Returns a file descriptor for the new socket, or -1 for errors.  */
extern int socket (int __domain, int __type, int __protocol) __THROW;
//参数：协议族 服务类型 通常0
```

## 命名socket

将一个socket与socket地址绑定称为给socket命名。一般服务器要命名socket，因为只有命名后客户端才能知道该如何连接它。客户端则通常不需要命名，而是采用匿名方式，即使用操作系统自动分配的socket地址。

```c++
/* Give the socket FD the local address ADDR (which is LEN bytes long).  */
# define __CONST_SOCKADDR_ARG	const struct sockaddr *
extern int bind (int __fd, __CONST_SOCKADDR_ARG __addr, socklen_t __len) __THROW;
```

bind成功返回1，失败返回-1，并设置errno，常见两种errno：

+ EACCES，被绑定端口是受保护的地址。
+ EADDRINUSE，被绑定的地址正在使用中，比如TIME_WAIT状态。

## 监听socket

socket被命名之后，还不能马上接受客户连接，我们需要使用如下系统调用来**创建一个监听队列**以存放待处理的客户连接。

```c++
/* Prepare to accept connections on socket FD.
   N connection requests will be queued before further requests are refused.
   Returns 0 on success, -1 for errors.  */
extern int listen (int __fd, int __n) __THROW;
```

n 表示内核监听队列的最大长度，监听队列的长度如果超过n，则不再受理新的客户连接。自内核版本2.2之后，它只表示处于完全连接状态的socket的上限。典型值为5。处于半连接状态的socket上限则由`/proc/sys/net/ipv4/tcp_max_syn_backlog`内核参数定义。

不过监听队列中完整连接的上限通常比n要大。

## 例子：

代码清单5-3：

```c++
const char* ip = argv[1];
int port = atoi( argv[2] );
int backlog = atoi( argv[3] );

//1 创建
int sock = socket( PF_INET, SOCK_STREAM, 0 );
assert( sock >= 0 );

struct sockaddr_in address;
bzero( &address, sizeof( address ) );
address.sin_family = AF_INET;
inet_pton( AF_INET, ip, &address.sin_addr );
address.sin_port = htons( port );

//2 命名
int ret = bind( sock, ( struct sockaddr* )&address, sizeof( address ) );
assert( ret != -1 );

//3 监听
ret = listen( sock, backlog );
assert( ret != -1 );

//...

close(sock);
```

## 接受连接

下面系统调用从listen监听队列中接收一个连接：

```c++

```

接口：

```c++
/* Await a connection on socket FD.
   When a connection arrives, open a new socket to communicate with it,
   set *ADDR (which is *ADDR_LEN bytes long) to the address of the connecting
   peer and *ADDR_LEN to the address's actual length, and return the
   new socket's descriptor, or -1 for errors.

   This function is a cancellation point and therefore not marked with
   __THROW.  */
typedef unsigned int socklen_t
# define __SOCKADDR_ARG		struct sockaddr *
extern int accept (int __fd, __SOCKADDR_ARG __addr, socklen_t * __addr_len);
//参数：监听socket，获取远端socket地址，远端socket的长度由addrlen参数指出
```

accept只是从监听队列中取出连接，而不论连接处于何种状态（ESTABLISHED和CLOSE_WAIT）。更不关心任何网络状态的变化。

## 发起连接（客户端）

```c++
/* Open a connection on socket FD to peer at ADDR (which LEN bytes long).
   For connectionless socket types, just set the default address to send to
   and the only address from which to accept transmissions.
   Return 0 on success, -1 for errors.  */
# define __CONST_SOCKADDR_ARG	const struct sockaddr *
extern int connect (int __fd, __CONST_SOCKADDR_ARG __addr, socklen_t __len);
```

## 关闭连接

```c++
#inlcude <unistd.h>

int close(int fd);
```





