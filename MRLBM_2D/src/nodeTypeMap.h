#ifndef NODETYPEMAP_H
#define NODETYPEMAP_H

/*
+----+----+
|  4 |  8 |
+----+----+
|  1 |  2 |
+----+----+
*/

#define BULK (15) // fluid
#define SOLID (0) // solid

// Edges
#define NORTH (3)
#define SOUTH (12)
#define EAST (5)
#define WEST (10)

// Corners
#define NORTH_EAST (1)
#define NORTH_WEST (2)
#define SOUTH_EAST (4)
#define SOUTH_WEST (8)

#define MISSING_DEFINITION (0b11111111111111111111111111111111)

#endif